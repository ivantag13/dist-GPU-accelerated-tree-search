#ifndef GPU_UTIL_H
#define GPU_UTIL_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

    void gpu_info();

#ifdef __cplusplus
}
#endif

#endif // GPU_UTIL_H
