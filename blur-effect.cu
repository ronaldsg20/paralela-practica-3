
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

  __device__ void aplyBlur(int &x, int &y, int &kernel, int &w, int &h, int *input, int *output){
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
            if(0<=i && i<width-1 && 0<=j && j<height-1){
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
     
    int tn = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    int ini = (int)(width/totalThreads)*(tn-1);
    int fin = (int)(width/totalThreads)+ini;
    for (int i = ini; i < fin; i++)
    {
        for (int j = 0; j < height; j++)
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
     int h_threads, h_kernel,h_width,h_height;
     int *d_threads;
     int *d_kernel;
     int *d_width;
     int *d_height;

     Mat output;
     Mat input;

     //read parameters
     if ( argc != 5 )
    {
        printf("usage: ./blur-effect <Image_Path> <Image_out_Path> <KERNEL> <THREADS X BLOCK> <BLOCKS>\n");
        return -1;
    }
    h_kernel = atoi(argv[3]);
    int threadsXblock = atoi(argv[4]);
    int blocks = atoi(argv[4]);
    h_threads = threadsXblock* blocks;
    String oFile = argv[2];

    //read the image and set width and height
    input = imread( argv[1], IMREAD_COLOR );
    if ( !input.data )
    {
        printf("No image data \n");
        return -1;
    }
    width = input.cols;
    height =input.rows;
    // define the output as a clone of input image
    output = input.clone();

    int *d_input;
    int *d_output;
    int *h_input;
    int *h_output;

     // malloc and cudaMalloc
     cudaMalloc(d_height,sizeof(int));
     cudaMalloc(d_kernel,sizeof(int));
     cudaMalloc(d_width,sizeof(int));
     cudaMalloc(d_threads,sizeof(int));

     cudaMalloc(&d_input,width*height*sizeof(int)*3);
     cudaMalloc(&d_output,width*height*sizeof(int)*3);
     
     malloc(&h_input,width*height*sizeof(int)*3);
     malloc(&h_output,width*height*sizeof(int)*3);

     // set initial values
     Vec3b pixel;

     for(int i=0;i<width;i++){
       for(int j=0;j<height;j++){
        pixel = input.at<Vec3b>(Point(i,j));
        h_input[(j*width*3)+(i*3)+0]= pixel.val(0);
        h_input[(j*width*3)+(i*3)+1]= pixel.val(1);
        h_input[(j*width*3)+(i*3)+2]= pixel.val(2);
       }
     }

     // MemCpy: host to device

     cudaMemcpy(d_input, h_input, sizeof(int)*width*height*3, cudaMemcpyHostToDevice);
     cudaMemcpy(d_kernel, h_kernel, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_threads, h_threads, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_width, h_width, sizeof(int), cudaMemcpyHostToDevice);
     cudaMemcpy(d_height, h_height, sizeof(int), cudaMemcpyHostToDevice);

     // define blocks 

     // Launch kernel 
     
     blur<<<blocks,threadsXblock>>>(d_input,d_output, d_kernel, d_threads, d_width, d_height);

     // MemCpy: device to host
     cudaMemcpy(h_output, d_output, sizeof(int)*width*height*3, cudaMemcpyDeviceToHost);

     for(int i=0;i<width;i++){
       for(int j=0;j<height;j++){
        
        pixel = Vec3b(h_output[(j*width*3)+(i*3)+0],h_output[(j*width*3)+(i*3)+1], h_output[(j*width*3)+(i*3)+2]);
        output.at<Vec3b>(Point(i,j))= pixel;
       }
     }

     // save data
     imwrite( oFile, output );

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
 
 