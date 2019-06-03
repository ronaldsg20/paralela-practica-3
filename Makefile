all:
	nvcc blur-effect.cu `pkg-config opencv --cflags --libs opencv` -I "/usr/local/cuda/samples/common/inc" -std=c++11 -o blur-effect