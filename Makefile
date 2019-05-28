all:
/usr/local/cuda-8.0/bin/nvcc -I /usr/local/cuda/samples/common/inc/ `pkg-config opencv --cflags --libs` blur-effect.cu `pkg-config --cflags opencv` -o blur-effect