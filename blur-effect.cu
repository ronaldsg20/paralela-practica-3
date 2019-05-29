
/**
 * Blur-effect
 */

 #include <stdio.h>
 #include <math.h>
 // For the CUDA runtime routines (prefixed with "cuda_")
 #include <cuda_runtime.h>

 #include <stdlib.h>
 #include <cstdint>
 #include <opencv2/opencv.hpp>
 
  using namespace cv;
  using namespace std;
  

// function aviable only on the device

  __device__ void aplyBlur(const int &x, const int &y, int &kernel, const int &w, const int &h, int *input, int *output){
    // collect the average data of neighbours 
    int blue,green,red;
    blue=green=red=0;
    int n=0;
    int pixel_pos;

    for(int i = x - (kernel/2); i < x+(kernel/2); i++)
    {    
        for (int j = y-(kernel/2); j < y+(kernel/2); j++)
        {
            //check if the point is in the image limits
            if(0<=i && i<w-1 && 0<=j && j<h-1){
                pixel_pos = (i*w*3)+(j*3);
                blue += input[pixel_pos+0];
                green += input[pixel_pos+1];
                red += input[pixel_pos+2];
                n++;
            }
        }
    }
    
    if(n!=0){
         //write the average on the output image
        output[pixel_pos+0]=blue/n;
        output[pixel_pos+1]=green/n;
        output[pixel_pos+2]=red/n;
    }
   
}

 /**
  * CUDA Kernel Device code
  * 
  */ 
 /*****************************************************************************/
 
 __global__ void blur(int *input,int *output, int *kernel, int *totalThreads, int *width, int *height)
 {   
     
    int tn = (blockDim.x * blockIdx.x) + threadIdx.x+1;
    
    int ini = (int)((int)*width/(int)*totalThreads)*(tn);
    int fin = (int)((int)*width/(int)*totalThreads)+ini;
    for (int i = ini; i < fin; i++)
    {
        for (int j = 0; j < (int)*height; j++)
        {
            aplyBlur(i,j,*kernel, *width,*height,input, output);
        }
    }
     
 }
 
 
 /******************************************************************************
  * Host main routine
  */
 int main(int argc, char **argv)
 {   
     // define variables
     int h_threads;
     int h_kernel;
     int h_width;
     int h_height;

     int *d_threads;
     int *d_kernel;
     int *d_width;
     int *d_height;

     Mat output;
     Mat input;
    // handle errors

    cudaError_t error = cudaSuccess;
     //read parameters
     if ( argc != 6 )
    {
        printf("usage: ./blur-effect <Image_Path> <Image_out_Path> <KERNEL> <THREADS X BLOCK> <BLOCKS>\n");
        return -1;
    }
    h_kernel = atoi(argv[3]);
    int threadsXblock = atoi(argv[4]);
    int blocks = atoi(argv[5]);
    h_threads = threadsXblock* blocks;
    String oFile = argv[2];

    //read the image and set width and height
    input = imread( argv[1], IMREAD_COLOR );
    if ( !input.data )
    {
        printf("No image data \n");
        return -1;
    }
    h_width = input.cols;
    h_height =input.rows;
    // define the output as a clone of input image
    output = input.clone();
    //imwrite( oFile, output ); // just for test

    printf(" Processing image %s \n width: %d  - Heigh : %d \n",argv[1],h_width,h_height);

    printf("Kernel : %d   Threads x Block: %d   # of Blocks: %d \n",h_kernel,threadsXblock,blocks);

    int *d_input;
    int *d_output;
    int *h_input;
    int *h_output;

     // malloc and cudaMalloc
     error = cudaMalloc(&d_height,sizeof(int));
     if (error != cudaSuccess){
        fprintf(stderr, "Failed to allocate mem for d_height (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
     error = cudaMalloc(&d_kernel,sizeof(int));
     if (error != cudaSuccess){
        fprintf(stderr, "Failed to allocate mem for d_kernel (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
     error = cudaMalloc(&d_width,sizeof(int));
     if (error != cudaSuccess){
        fprintf(stderr, "Failed to allocate mem for d_width (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
     error = cudaMalloc(&d_threads,sizeof(int));
     if (error != cudaSuccess){
        fprintf(stderr, "Failed to allocate mem for d_threads (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

     error = cudaMalloc(&d_input,h_width*h_height*sizeof(int)*3);
     if (error != cudaSuccess){
        fprintf(stderr, "Failed to allocate mem for d_input (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
     error = cudaMalloc(&d_output,h_width*h_height*sizeof(int)*3);
     if (error != cudaSuccess){
        fprintf(stderr, "Failed to allocate mem for d_output (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


     size_t size = h_width * h_height * 3 * sizeof(int);
     h_input = (int *)malloc(size);
     h_output = (int *)malloc(size);

     // set initial values
     Vec3b pixel;
     //pixel = input.at<Vec3b>(Point(i,j));
     printf("ORIGINAL IMAGE \n");
     for(int i=0;i<h_width;i++){
       for(int j=0;j<h_height;j++){
        
        h_input[(j*h_width*3)+(i*3)+0]= input.at<Vec3b>(i,j)[0];
        h_input[(j*h_width*3)+(i*3)+1]= input.at<Vec3b>(i,j)[1];
        h_input[(j*h_width*3)+(i*3)+2]= input.at<Vec3b>(i,j)[2];
       }
     }

     for(int t =0;t<20;t++){
         for (int k=0;k<20;k++){
            printf("[%d , %d , %d] ",h_input[(k*h_width*3)+(t*3)+0],h_input[(k*h_width*3)+(t*3)+1], h_input[(k*h_width*3)+(t*3)+2]);
         }
         printf("\n");
     }

     // MemCpy: host to device

     error = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
     if (error != cudaSuccess){
        fprintf(stderr, "Failed to copy on device (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_kernel, &h_kernel, sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        fprintf(stderr, "Failed to  to copy on device(error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_threads, &h_threads, sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        fprintf(stderr, "Failed to  to copy on device (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_width, &h_width, sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        fprintf(stderr, "Failed to  to copy on device (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    error = cudaMemcpy(d_height, &h_height, sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        fprintf(stderr, "Failed to  to copy on device (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    printf("CudaMemcpy host to device done.\n");

     // Launch kernel 
     
     blur<<<blocks,threadsXblock>>>(d_input,d_output, d_kernel, d_threads, d_width, d_height);

     //last error
     error = cudaGetLastError();
    if (error != cudaSuccess){
        fprintf(stderr, "Failed to launch blur (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

     // MemCpy: device to host
     error = cudaMemcpy(h_output, d_output, sizeof(int)*h_width*h_height*3, cudaMemcpyDeviceToHost);
     if (error != cudaSuccess){
        fprintf(stderr, "Failed to  to copy from device (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    printf("MODIFIED IMAGE \n");

    for(int t =0;t<20;t++){
        for (int k=0;k<20;k++){
           printf("[%d , %d , %d] ",h_output[(k*h_width*3)+(t*3)+0],h_output[(k*h_width*3)+(t*3)+1], h_output[(k*h_width*3)+(t*3)+2]);
        }
        printf("\n");
    }
     for(int i=0;i<h_width;i++){
       for(int j=0;j<h_height;j++){
        pixel = Vec3b(h_output[(j*h_width*3)+(i*3)+0],h_output[(j*h_width*3)+(i*3)+1], h_output[(j*h_width*3)+(i*3)+2]);
        output.at<Vec3b>(Point(i,j))= pixel;
       }
     }

     // save data
     //imwrite( oFile, output );

     // free memory

     cudaFree(d_height);
     cudaFree(d_width);
     cudaFree(d_output);
     cudaFree(d_input);
     cudaFree(d_kernel);
     cudaFree(d_threads);

     free(h_input);
     free(h_output);

     return 0;
 }
 
 