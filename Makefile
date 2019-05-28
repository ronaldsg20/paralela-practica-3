all:
/usr/local/cuda-8.0/bin/nvcc -I /usr/local/cuda/samples/common/inc/ blur-effect.cu -o blur-effect