#ifndef GPU_UTIL_H
#define GPU_UTIL_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

    __host__ void gpu_info();

#ifdef __cplusplus
}
#endif

#endif // GPU_UTIL_H
