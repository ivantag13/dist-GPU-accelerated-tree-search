#include "gpu_util.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

__host__ void gpu_info()
{
    struct cudaDeviceProp prop;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    printf("Device name:          %s\n", prop.name);
    printf("SM count:             %d\n", prop.multiProcessorCount);
    // printf("CUDA cores/SM (est):  %d\n", e.g. 64);
    printf("Clock rate (kHz):     %d\n", prop.clockRate);
    // printf("Memory bandwidth (GB/s): ~%.1f\n",look up or use prop.memoryClockRate & busWidth );
    return;
}