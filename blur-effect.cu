#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <errno.h>

#pragma pack(push,1)
typedef struct {
  char         filetype[2];   /* magic - always 'B' 'M' */
  unsigned int filesize;
  short        reserved1;
  short        reserved2;
  unsigned int dataoffset;    /* offset in bytes to actual bitmap data */
} file_header;

typedef struct {
  file_header  fileheader;
  unsigned int headersize;
  int          width;
  int          height;
  short        planes;
  short        bitsperpixel;  /* we only support the value 24 here */
  unsigned int compression;   /* we do not support compression */
  unsigned int bitmapsize;
  int          horizontalres;
  int          verticalres;
  unsigned int numcolors;
  unsigned int importantcolors;
} bitmap_header;
#pragma pack(pop)

readImage(){
    FILE *fp;
    FILE *fp, *out;
  bitmap_header* hp;
  int n, x, xx, y, yy, ile, avgR, avgB, avgG, B, G, R;
  unsigned char *data;
  int rc, i, blurSize = kernel;
  pthread_t thread[threads];
  thread_data thrdata[threads];

  //Open input file:
  fp = fopen(input, "r");
  if(fp == NULL){
    //cleanup
  }

  //Read the input file headers:
  hp = (bitmap_header*) malloc(sizeof(bitmap_header));
  if(hp == NULL)
    return 3;

  n = fread(hp, sizeof(bitmap_header), 1, fp);
  if(n < 1){
    //cleanup
  }
  //Read the data of the image:
  data = (char*) malloc(sizeof(char) * hp->bitmapsize);
  if(data == NULL){
    //cleanup
  }

  fseek(fp, sizeof(char) * hp->fileheader.dataoffset, SEEK_SET);
  n = fread(data, sizeof(char), hp->bitmapsize, fp);
  if(n < 1){
    //cleanup
  }

}
int main(){

    return 0;
}