all:
	/usr/local/cuda-8.0/bin/nvcc blur-effect.cu `pkg-config opencv --cflags --libs opencv` -o blur-effect