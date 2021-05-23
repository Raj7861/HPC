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


#include <helper_functions.h>    


#include <helper_cuda.h>        

#define MAX_EPSILON_ERROR 5e-3f

#define tile_width 8

const char *sampleName = "simpleTexture";

__constant__ float const_sharp[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
__constant__ float const_edge[9] = {-1,0,1,-2,0,2,-1,0,1};
__constant__ float const_av[9] = {1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9};

__global__ void conv_const(float *out,float *matrix,int width,int height,int m_width,int m_height){
    
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
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
                value = value + matrix[(j-(int)(m_height/2)+p)*width+(i-(int)(m_width/2)+l)]*const_edge[p*m_width+l];
            }
        }
    }
    out[j*width+i] = value;
}


void runTest(int argc, char **argv, char *imageFilename);


int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    char image_name[5][1024] = {"image21.pgm","lena_bw.pgm","man.pgm","mandrill.pgm","ref_rotated.pgm"};

    for(int i=0;i<5;i++){
        runTest(argc, argv,image_name[i]);
    }
    cudaDeviceReset();
    return 0;
}

void runTest(int argc, char **argv, char *imageFilename)
{

    int devID = findCudaDevice(argc, (const char **) argv);

    
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
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);


    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    

    float *gData = 0,*dData=0;
    checkCudaErrors(cudaMalloc((void **) &gData, size));
    checkCudaErrors(cudaMalloc((void **) &dData, size));
    cudaMemcpy(dData,hData, size, cudaMemcpyHostToDevice);

    StopWatchInterface *g_timer = NULL;
    sdkCreateTimer(&g_timer);
    sdkStartTimer(&g_timer);

    conv_const<<<dimGrid, dimBlock,0>>>(gData,dData,width,height,3,3); 

    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStopTimer(&g_timer);
    printf("Processing time for global: %f (ms)\n", sdkGetTimerValue(&g_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&g_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&g_timer);

    
    float *global_out = (float *) malloc(size);
    checkCudaErrors(cudaMemcpy(global_out,gData,size,cudaMemcpyDeviceToHost));


    char global_outputfile[1024];
    strcpy(global_outputfile, imagePath);
    strcpy(global_outputfile + strlen(imagePath) - 4, "_global_out.pgm");
    sdkSavePGM(global_outputfile, global_out, width, height);
    printf("Wrote '%s'\n", global_outputfile);

    free(imagePath);
    
}
