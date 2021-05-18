#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct ImageSegment
{
	cv::Mat img_;
	cv::Rect rect_;

	std::string text_;
	float score_ = 0;
};

struct ImageList
{
	std::vector<ImageSegment> segments_;
	int maxWidth_ = 0;
	int modelHeight_ = 0;
};

inline float ycenter(const cv::Rect &r)
{
	return r.y + r.height / 2.0;
}

const cv::Rect &adjust(cv::Rect &r, float margin)
{
	r.x -= margin;
	r.y -= margin;
	r.width += 2 * margin;
	r.height += 2 * margin;

	return r;
}

int CanvasSize = 1280; // detector image size
float MagRatio = 1.0;  // default magnifier ratio
float TextThreshold = 0.5;
float TextThresholdLow = 0.4;
float LinkThreshold = 0.4;

int ModelHeight = 64;

struct Engine
{
	torch::jit::Module det_;
	torch::jit::Module reco_;
	torch::Device device_;
	std::vector<std::string> characters_;

	Engine(torch::Device device) : device_(device) {}

	int loadModels(const std::string &det, const std::string &reco);

	std::vector<cv::Rect> getTextBoxes(cv::Mat img);
	static std::vector<cv::Rect> mergeBoxes(std::vector<cv::Rect> boxes, float heightThres = 0.5, float posThres = 0.5, float widthThres = 1.0, float margin = 0.05);
	static ImageList getImageList(cv::Mat img, const std::vector<cv::Rect> &boxes, int modelHeight);

	int recognize(ImageList &, int modelHeight);
	std::string decode(torch::Tensor index);
};

int Engine::loadModels(const std::string &det, const std::string &reco)
{
	this->det_ = torch::jit::load(det);
	this->reco_ = torch::jit::load(reco);

	this->det_.eval();
	this->reco_.eval();

	this->det_.to(this->device_);
	this->reco_.to(this->device_);

	return 0;
}

std::vector<cv::Rect> Engine::getTextBoxes(cv::Mat img)
{
	// resize
	float targetSize = std::max(img.cols, img.rows) * MagRatio;
	if (targetSize > CanvasSize)
		targetSize = CanvasSize;

	float ratio = targetSize / std::max(img.cols, img.rows);
	int width = int(img.cols * ratio), height = int(img.rows * ratio);

	// image for detection model
	cv::Mat nimg;
	cv::resize(img, nimg, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

	// pad to 32 pix
	int width32 = width, height32 = height;
	if (width32 % 32 != 0)
		width32 += (32 - width32 % 32);
	if (height32 % 32 != 0)
		height32 += (32 - height32 % 32);

	cv::Mat canvas(cv::Size(width32, height32), CV_8UC3);
	nimg.copyTo(canvas(cv::Rect(0, 0, width, height)));
	canvas.convertTo(nimg, CV_32FC3, 1 / 255.0);

	// convert to tensor
	const int kCHANNELS = 3;
	auto input = torch::from_blob(nimg.ptr<float>(), {1, nimg.rows, nimg.cols, kCHANNELS}, at::kFloat).clone();
	input = input.permute({0, 3, 1, 2}); // to [b, c, h, w]

	// normalize
	input[0][0] = input[0][0].sub(0.485).div(0.229);
	input[0][1] = input[0][1].sub(0.456).div(0.224);
	input[0][2] = input[0][2].sub(0.406).div(0.225);

	// move to device and run net
	input = input.to(this->device_);
	auto score = this->det_.forward({input}).toTuple()->elements()[0].toTensor().to(torch::kCPU).squeeze();

	std::cout << "image: " << nimg.cols << "x" << img.rows << ", " << ratio << std::endl;
	std::cout << "input: " << input.sizes() << std::endl;
	std::cout << "result: " << score.sizes() << std::endl;

	score = score.to(torch::kCPU).squeeze();
	//	score = score.squeeze().detach();
	auto text = score.select(2, 0).clone();
	auto link = score.select(2, 1).clone();

	std::cout << "copied." << std::endl;

	// copy to opencv
	cv::Mat textmap(text.size(0), text.size(1), CV_32FC1);
	std::memcpy(textmap.data, text.data_ptr(), sizeof(float) * text.numel());

	cv::Mat linkmap(text.size(0), text.size(1), CV_32FC1);
	std::memcpy(linkmap.data, link.data_ptr(), sizeof(float) * link.numel());

	// binarization
	cv::Mat textscore, linkscore;
	cv::threshold(textmap, textscore, TextThresholdLow, 1.0, cv::THRESH_BINARY);
	cv::threshold(linkmap, linkscore, LinkThreshold, 1.0, cv::THRESH_BINARY);

	// combine link + text
	cv::Mat tscore = textscore + linkscore;
	tscore.setTo(1.0, tscore > 1.0);
	tscore.convertTo(tscore, CV_8UC1);

	cv::Mat labels, stats, centroids;
	int connectivity = 4;
	cv::connectedComponentsWithStats(tscore, labels, stats, centroids, connectivity);

	std::cout << stats.rows << ", " << stats.cols << std::endl;

	// link only mask
	auto tlink = linkscore.clone();
	tlink.setTo(0, (textscore == 1)); //unmask text area
	auto linkmask = tlink == 1;

	auto iw = textmap.cols, ih = textmap.rows;
	float upscale = float(nimg.cols) / iw / ratio;

	// find boxes
	std::vector<cv::Rect> boxes;
	for (int i = 0; i < stats.rows; i++)
	{
		int x = stats.at<int>(cv::Point(0, i));
		int y = stats.at<int>(cv::Point(1, i));
		int w = stats.at<int>(cv::Point(2, i));
		int h = stats.at<int>(cv::Point(3, i));
		int area = stats.at<int>(cv::Point(4, i));

		if (area < 10)
		{
			continue;
		}

		auto mask = labels == i;
		double minv = 0, maxv = 0;
		cv::minMaxLoc(textmap, &minv, &maxv, 0, 0, mask);

		if (maxv < TextThreshold)
		{
			std::cout << "Skipping(maxv) " << i << std::endl;
			continue;
		}

		cv::Mat segmap(textmap.size(), CV_8UC1);
		segmap.setTo(255, mask);
		segmap.setTo(0, linkmask); // remove link only area

		int pad = int(sqrt(area * std::min(w, h) / (w * h)) * 2);
		int sx = std::max(0, x - pad),
			ex = std::min(iw, x + w + pad),
			sy = std::max(0, y - pad),
			ey = std::min(ih, y + h + pad);

		auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(pad + 1, pad + 1));
		auto patch = segmap.colRange(sx, ex).rowRange(sy, ey);
		cv::Mat dpatch;
		cv::dilate(patch, dpatch, kernel);
		dpatch.copyTo(patch);

		// find bbox
		cv::Mat idx;
		std::vector<cv::Point2f> pts(4);
		cv::findNonZero(segmap, idx);
		auto rect = cv::minAreaRect(idx);
		rect.points(pts.data());

		// convert to rect
		auto x1 = int(std::min({pts[0].x, pts[1].x, pts[2].x, pts[3].x}) * upscale);
		auto y1 = int(std::min({pts[0].y, pts[1].y, pts[2].y, pts[3].y}) * upscale);
		auto x2 = int(std::max({pts[0].x, pts[1].x, pts[2].x, pts[3].x}) * upscale);
		auto y2 = int(std::max({pts[0].y, pts[1].y, pts[2].y, pts[3].y}) * upscale);
		boxes.emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
	}

	std::cout << text.sizes() << "," << link.sizes() << ", " << boxes.size() << std::endl;

	std::sort(boxes.begin(), boxes.end(), [](const auto &p1, const auto &p2) { return ycenter(p1) < ycenter(p2) || (ycenter(p1) == ycenter(p2) && p1.x < p2.x); });
	auto nboxes = mergeBoxes(boxes);

	cv::cvtColor(nimg, nimg, cv::COLOR_BGR2RGB);
	cv::Scalar color(0.5, 0.5, 0);
	for (auto box : nboxes)
	{
		box.x = int(box.x * ratio);
		box.y = int(box.y * ratio);
		box.width = int(box.width * ratio);
		box.height = int(box.height * ratio);
		cv::rectangle(nimg, box, color);
	}

//	cv::imshow("Display", textscore);
//	cv::waitKey(0);


//	cv::imshow("Display", nimg);
//	cv::waitKey(0);

	return nboxes;
}

std::string Number = "0123456789";
std::string Symbol = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
std::string Eng = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

std::vector<std::string> loadCharacters(const std::string &file)
{
	std::vector<std::string> dict{"[blank]"};

	for (char c : Number + Symbol + Eng)
	{
		dict.emplace_back(std::string(1, c));
	}

	std::ifstream ifs(file);
	std::string s;
	while (std::getline(ifs, s))
	{
		dict.emplace_back(s);
	}

	return dict;
}

int main(int argc, const char *argv[])
{
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Working on GPU." : "Working on CPU.") << "\n";

	Engine eng(device);
	eng.loadModels("craft.pt", "crnn-zh.pt");
	eng.characters_ = loadCharacters("ch_sim_char.txt");

	// model.det_.dump(true, false, false);
	// model.reco_.dump(true, false, false);

	// load image
	std::string file = argv[1];
	auto img = cv::imread(file);
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	auto gray = cv::imread(file, cv::IMREAD_GRAYSCALE);

	// find and merge boxes
	auto boxes = eng.getTextBoxes(img);

	// recognize segments
	auto imgs = Engine::getImageList(gray, boxes, ModelHeight);
	std::cout << "Total " << imgs.segments_.size() << " segments" << std::endl;
	eng.recognize(imgs, ModelHeight);

	return 0;
}

template <typename T>
T mean(const std::vector<T> &v)
{
	T sum = std::accumulate(v.begin(), v.end(), T(0));
	return sum / v.size();
}

std::vector<cv::Rect> Engine::mergeBoxes(std::vector<cv::Rect> boxes, float heightThres, float posThres, float widthThres, float margin)
{
	std::vector<float> ycenters;
	std::vector<float> heights;

	std::vector<std::vector<cv::Rect>> blocks;

	std::vector<cv::Rect> block;
	for (auto const &box : boxes)
	{
		auto ypos = ycenter(box);
		if (block.empty())
		{
			block.emplace_back(box);
			ycenters.emplace_back(ypos);
			heights.emplace_back(box.height);
		}
		else
		{
			auto ymean = mean(ycenters), hmean = mean(heights);
			bool sameblock = (std::abs(box.height - hmean) < hmean * heightThres && std::abs(ypos - ymean) < hmean * posThres);
			if (sameblock)
			{
				block.emplace_back(box);
				ycenters.emplace_back(ypos);
				heights.emplace_back(box.height);
			}
			else
			{
				blocks.emplace_back(std::move(block));
				block = std::vector<cv::Rect>{box};
				ycenters = std::vector<float>{ypos};
				heights = std::vector<float>{float(box.height)};
			}
		}
	}
	if (!block.empty())
		blocks.emplace_back(block);

	std::vector<cv::Rect> nboxes;
	for (auto &block : blocks)
	{
		if (block.size() == 1)
		{
			auto box = block.front();
			auto m = int(margin * box.height);
			nboxes.emplace_back(adjust(box, m));
		}
		else
		{
			std::sort(block.begin(), block.end(), [](const cv::Rect &p1, const cv::Rect &p2) { return p1.x < p2.x; });

			int xmax = 0;
			std::vector<cv::Rect> active;
			std::vector<std::vector<cv::Rect>> merged;
			for (auto const &box : block)
			{
				if (active.empty())
				{
					active.emplace_back(box);
					xmax = box.x + box.width;
				}
				else
				{
					bool merge = std::abs(box.x - xmax) < widthThres * box.height;
					xmax = box.x + box.width;
					if (merge)
					{
						active.emplace_back(box);
					}
					else
					{
						merged.emplace_back(active);
						active = std::vector<cv::Rect>{box};
					}
				}
			}
			if (!active.empty())
				merged.emplace_back(active);

			cv::Rect nbox;
			for (auto const &block : merged)
			{
				if (block.size() == 1)
				{
					nbox = block.front();
				}
				else
				{
					const auto &box = block.front();
					int xmin = box.x, xmax = box.x + box.width;
					int ymin = box.y, ymax = box.y + box.height;
					for (size_t i = 1; i < block.size(); i++)
					{
						const auto &box = block[i];
						int x1 = box.x, x2 = box.x + box.width;
						int y1 = box.y, y2 = box.y + box.height;
						if (xmin > x1)
							xmin = x1;
						if (ymin > y1)
							ymin = y1;
						if (xmax < x2)
							xmax = x2;
						if (ymax < y2)
							ymax = y2;
					}

					nbox.x = xmin;
					nbox.y = ymin;
					nbox.width = xmax - xmin;
					nbox.height = ymax - ymin;
				}

				auto m = int(margin * nbox.height);
				nboxes.emplace_back(adjust(nbox, m));
			}
		}
	}

	return nboxes;
}

ImageList Engine::getImageList(cv::Mat img, const std::vector<cv::Rect> &boxes, int modelHeight)
{
	ImageList imgs;
	float max_ratio = 1.0;
	for (auto const &box : boxes)
	{
		auto seg = img(box);
		auto ratio = float(box.width) / box.height;
		int width = int(ratio * ModelHeight);

		cv::Mat nseg;
		cv::resize(seg, nseg, cv::Size(width, modelHeight), 0, 0, cv::INTER_AREA);

		imgs.segments_.emplace_back(ImageSegment{nseg, box});
		max_ratio = std::max(ratio, max_ratio);
	}

	imgs.maxWidth_ = int(ceil(max_ratio) * modelHeight);
	imgs.modelHeight_ = modelHeight;
	return std::move(imgs);
}

class ImageDataset : public torch::data::Dataset<ImageDataset>
{
private:
	ImageList &images_;

public:
	ImageDataset(ImageList &images) : images_(images) {}

	torch::data::Example<> get(size_t index) override
	{
		torch::Tensor tensor = torch::zeros({1});

		auto &seg = images_.segments_[index];
		cv::Mat nimg;
		seg.img_.convertTo(nimg, CV_32FC1, 1 / 255.0);

		//normalize
		nimg -= 0.5;
		nimg /= 0.5;

		//pad
		assert(images_.modelHeight_ == nimg.rows);
		cv::Mat nseg(images_.modelHeight_, images_.maxWidth_, CV_32FC1);
		nseg = 1.0;
		nimg.copyTo(nseg(cv::Rect(0, 0, nimg.cols, nimg.rows)));

		auto input = torch::from_blob(nseg.ptr<float>(), {1, nseg.rows, nseg.cols}, at::kFloat).clone();
		return {input, tensor};
	}

	torch::optional<size_t> size() const override
	{
		return images_.segments_.size();
	};
};

std::vector<std::string> LatinCharacters = {
	"[blank]", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "\"", "#", "$", "%", "&", "'",
	"(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^",
	"_", "`", "{", "|", "}", "~", " ",
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
	"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
	"À", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Í", "Î", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "Ø", "Ú", "Û", "Ü", "Ý",
	"Þ", "ß", "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï", "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "ø",
	"ù", "ú", "û", "ü", "ý", "þ", "ÿ", "ą", "ę", "Į", "į", "ı", "Ł", "ł", "Œ", "œ", "Š", "š", "ų", "Ž", "ž"};

std::string Engine::decode(torch::Tensor index)
{
	std::string text;
	for (int i = 0; i < index.size(0); i++)
	{
		auto idx = index[i].item<int>();
		if (idx == 0)
			continue;

		if (i > 0 && idx == index[i - 1].item<int>())
			continue;

		text += this->characters_[idx];
	}

	return text;
}

struct BeamEntry
{
	float probBlank_ = 0.0;
	float probNonBlank_ = 0.0;

	std::vector<int> prefix_;
};

//TODO: 
#if 0
BeamEntry& getBeamEntry(std::vector<BeamEntry>& beam, const std::vector<int>& prefix)
{
	auto it = std::find_if(beam.begin(), beam.end(), [](auto& e) { return e.prefix_ == prefix});
	if (it != beam.end()) {
		return *it;
	}

	beam.emplace_back(BeamEntry{0.0, 0.0, prefix});
	return beam.back();
}

std::string Engine::decode(torch::Tensor probs, int beamSize)
{
	int T = probs.size(0), S = probs.size(1);

	std::vector<BeamEntry> beam{BeamEntry{}};
	for (int t=0; t < T; t++) {
		std::vector<BeamEntry> nbeam;

		for (int s=0; s < S; s++) {
			auto p = probs[t][s].item<float>();

			for (auto& entry: beam) {
				if (s == 0) { //blank
					auto &nentry = getBeamEntry(nbeam, entry.prefix);
				}
			}
		}
	}
}
#endif

int Engine::recognize(ImageList &imgs, int modelHeight)
{
	int maxWidth = imgs.maxWidth_;
	int batchMaxLen = int(maxWidth / 10.0 + 0.5);

	int batchSize = 10;

	torch::NoGradGuard no_grad;

	ImageDataset dataset(imgs);
	auto loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		dataset.map(
			torch::data::transforms::Stack<>()),
		batchSize);

	torch::Tensor text = torch::zeros({batchSize, batchMaxLen}).to(this->device_);
	int total = 0;
	for (auto &batch : *loader)
	{
		auto input = batch.data.to(this->device_);
		auto preds = this->reco_.forward({input, text}).toTensor();

		std::cout << input.sizes() << "," << text.sizes() << "," << preds.sizes() << std::endl;
		namespace F = torch::nn::functional;
		auto probs = F::softmax(preds, F::SoftmaxFuncOptions(2)); // [batch, seq, class]
		std::cout << input.sizes() << "," << text.sizes() << "," << preds.sizes() << ", probs=" << probs.sizes() << std::endl;
		probs = probs.detach().to(torch::kCPU);

		// re-normalize
		// preds_prob[:,:,ignore_idx] = 0.
		// auto norm = probs.sum({2}).unsqueeze_(-1);
		// probs.div(norm);

		// greedy search
		auto t = probs.max(2);
		auto index = std::get<1>(t);

		int batchSize = input.size(0);
		for (int i = 0; i < batchSize; i++)
		{
			std::string text = this->decode(index[i]);
			imgs.segments_[total].text_ = text;
			std::cout << imgs.segments_[total].rect_ << text << std::endl;
			total++;
		}

		//		std::cout << input.sizes() << "," << text.sizes() << "," << preds.sizes() << ", probs=" << probs.sizes() << std::endl;
	}

	return 0;
}

