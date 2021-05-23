#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>   

// CUDA helper functions
#include <helper_cuda.h>         

#define MAX_EPSILON_ERROR 5e-3f

#define tile_width 8

char image_name[5][1024] = {"image21.pgm","lena_bw.pgm","man.pgm","mandrill.pgm","ref_rotated.pgm"};
const char *sampleName = "simpleTexture";

float* convolution(float *matrix,float *mask,int width,int height,int m_width,int m_height){
    float* out = (float*)malloc(sizeof(float)*width*height);
    for(int j = 0;j<height;++j){
        for(int i =0;i<width;++i){
            float value = 0;
            for(int p=0;p<m_height;++p){
                for(int l=0;l<m_width;++l){
                    if((j-(int)(m_height/2)+p)<0){
                        value = value + 0; 
                    }
                    else if((i-(int)(m_width/2)+l)>=width){
                        value = value + 0;
                    }
                    else if((j-(int)(m_height/2)+p)>=height){
                        value = value + 0;
                    }
                    else if((i-(int)(m_width/2)+l)<0){
                        value = value + 0;
                    }
                    else{
                        value = value + matrix[(j-(int)(m_height/2)+p)*width+(i-(int)(m_width/2)+l)]*mask[p*m_width+l];
                    }
                }
            }
            if(value>0.4 || value<-0.4){
                out[j*width+i] = 1;
            }
            else{
                out[j*width+i] = 0;
            }
            //out[j*width+i] = value;
        }
    }
    return out;
}

void execute(int argc, char **argv, char *imageFilename, int j);

int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    char ker[3][1024] = {"SHARPENING","EDGE_DETECTION","AVERAGING"};
    for(int j=0;j<3;j++){
        printf("%s \n",ker[j]);
    for(int i=0;i<5;i++){
    execute(argc, argv,image_name[i], j);
    printf("\n");
    }
 }

    //cudaDeviceReset();
    return 0;
}

void execute(int argc, char **argv, char *imageFilename, int j)
{

    //convulution mask
    float *sharpening = (float*)malloc(sizeof(float)*3*3);
    float *edgeDectection = (float*)malloc(sizeof(float)*3*3);
    float *averaging = (float*)malloc(sizeof(float)*3*3);
    float sharp[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
    float edge[9] = {-1,0,1,-2,0,2,-1,0,1};
    float av[9] = {1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9};
    sharpening=&sharp[0];
    edgeDectection=&edge[0];
    averaging=&av[0];

    
    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    
    float *sData = NULL;

    StopWatchInterface *s_timer = NULL;
    sdkCreateTimer(&s_timer);
    sdkStartTimer(&s_timer);

    //execute serial 
    if(j==0)
    sData = convolution(hData,sharpening,width,height,3,3);
    else if(j==1)
        sData = convolution(hData,edgeDectection,width,height,3,3);
    else
        sData = convolution(hData,averaging,width,height,3,3);
    
    sdkStopTimer(&s_timer);
    printf("Processing time for serial: %f (ms)\n", sdkGetTimerValue(&s_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&s_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&s_timer);

     //Write result to file
    char serial_outputfile[1024];
    strcpy(serial_outputfile, imagePath);
    strcpy(serial_outputfile + strlen(imagePath) - 4, "_serial_out.pgm");
    sdkSavePGM(serial_outputfile, sData, width, height);
    printf("Wrote '%s'\n", serial_outputfile);

    free(imagePath);
    
}

