#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <math.h>


#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif


#include <cuda_runtime.h>


#include <helper_functions.h>    


#include <helper_cuda.h>         

#define MAX_EPSILON_ERROR 5e-3f

#define tile_width 8

char *imageFilename;
const char *refFilename   = "ref_rotated.pgm";

const char *sampleName = "simpleTexture";

texture<float, 2, cudaReadModeElementType> tex;
texture<float,2,cudaReadModeElementType> tex_sharp;
texture<float,2,cudaReadModeElementType> tex_edge;
texture<float,2,cudaReadModeElementType> tex_av;


__global__ void convolution_texture(float *out,int width,int height,int m_width,int m_height){
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
                 value = value + tex2D(tex,i-(int)(m_width/2)+l , j-(int)(m_height/2)+p)*tex2D(tex_sharp,l,p);
            }
        }
    }
    out[j*width+i] =value;

}

void runTest(int argc, char **argv,int n);

int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);


char image_name[5][1024] = {"image21.pgm","lena_bw.pgm","man.pgm","mandrill.pgm","ref_rotated.pgm"};  
for(int i = 3;i<10;i=i+2){
    for(int j=0;j<4;++j){
        imageFilename = image_name[j];
        runTest(argc, argv,i);
    }  
}

    cudaDeviceReset();
    return 0;
}

void runTest(int argc, char **argv,int n)
{

    int devID = findCudaDevice(argc, (const char **) argv);

    //convulution mask
    float *sharpening = (float*)malloc(sizeof(float)*3*3);
    float *edgeDectection = (float*)malloc(sizeof(float)*3*3);
    float *averaging = (float*)malloc(sizeof(float)*n*n);
    float sharp[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
    float edge[9] = {-1,0,1,-2,0,2,-1,0,1};

    sharpening=&sharp[0];
    edgeDectection=&edge[0];

    for(int i=0;i<n*n;++i){
        averaging[i] = 1/9;
    }
    
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

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc sharp_cd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc edge_cd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc av_cd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    cudaArray *sharp_cu;
    cudaArray *edge_cu;
    cudaArray *av_cu;
    checkCudaErrors(cudaMallocArray(&cuArray,&channelDesc,width,height));
    checkCudaErrors(cudaMallocArray(&sharp_cu,&sharp_cd,3,3));
    checkCudaErrors(cudaMallocArray(&edge_cu,&edge_cd,3,3));
    checkCudaErrors(cudaMallocArray(&av_cu,&av_cd,n,n));
    checkCudaErrors(cudaMemcpyToArray(cuArray,0,0,hData,size,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(sharp_cu,0,0,sharpening,3*3*sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(edge_cu,0,0,edgeDectection,3*3*sizeof(float),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(av_cu,0,0,averaging,n*n*sizeof(float),cudaMemcpyHostToDevice));

    
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    

    tex_sharp.addressMode[0] = cudaAddressModeWrap;
    tex_sharp.addressMode[1] = cudaAddressModeWrap;
    tex_sharp.filterMode = cudaFilterModeLinear;


    tex_av.addressMode[0] = cudaAddressModeWrap;
    tex_av.addressMode[1] = cudaAddressModeWrap;
    tex_av.filterMode = cudaFilterModeLinear;
    

    tex_edge.addressMode[0] = cudaAddressModeWrap;
    tex_edge.addressMode[1] = cudaAddressModeWrap;
    tex_edge.filterMode = cudaFilterModeLinear;
    


    
    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(tex_sharp, sharp_cu, sharp_cd));
    checkCudaErrors(cudaBindTextureToArray(tex_edge, edge_cu, edge_cd));
    checkCudaErrors(cudaBindTextureToArray(tex_av, av_cu, av_cd));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    
    float *txData = NULL;
    checkCudaErrors(cudaMalloc((void **) &txData, size));

    StopWatchInterface *t_timer = NULL;
    sdkCreateTimer(&t_timer);
    sdkStartTimer(&t_timer);

    convolution_texture<<<dimGrid,dimBlock,0>>>(txData,width,height,n,n);
    
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStopTimer(&t_timer);
    printf("Processing time for texture: %f (ms)\n", sdkGetTimerValue(&t_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&t_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&t_timer);

    
    float *tex_out = (float *) malloc(size);
    checkCudaErrors(cudaMemcpy(tex_out,txData,size,cudaMemcpyDeviceToHost));

    
    char number[1024];
    sprintf(number,"%d",n);
    strcat(number,"_texture_out.pgm");
    char tex_outputfile[1024];
    strcat(tex_outputfile, imagePath);
    strcpy(tex_outputfile + strlen(imagePath) - 4, number);
    sdkSavePGM(tex_outputfile, tex_out, width, height);
    printf("Wrote '%s'\n", tex_outputfile);

    free(imagePath);

}
