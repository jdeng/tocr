g++ -std=c++17 -I libtorch/include -I libtorch/include/torch/csrc/api/include -I/usr/include/opencv4 -L libtorch/lib -g -O2  -o tocr tocr.cpp  -Wl,--no-as-needed -ltorch_cpu  -ltorch_cuda  -lc10 -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lpthread

