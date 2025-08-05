#ifndef PFSP_STATISTIC_H
#define PFSP_STATISTIC_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdlib.h>
#include <stdio.h>

    /*******************************************************************************
    Data logging functions
    *******************************************************************************/

    void print_results_file_single_gpu(const int inst, const int lb, const int optimum, const int m, const int M,
                                       const unsigned long long int exploredTree, const unsigned long long int exploredSol,
                                       const double timer, double timeGpuCpy, double timeGpuMalloc, double timeGpuKer, double timeGenChild);

    void print_results_file_multi_gpu(
        const int inst, const int lb, const int D, int ws, const int optimum,
        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
        unsigned long long int *expTreeGPU, unsigned long long int *expSolGPU, unsigned long long int *genChildren,
        unsigned long long int *nStealsGPU, unsigned long long int *nSStealsGPU, unsigned long long int *nTerminationGPU,
        double *timeCudaMemCpy, double *timeCudaMalloc, double *timeKernelCall, double *timeIdle, double *timeTermination,
        double *timePoolOps, double *timeGenChildren);

    void print_results_file_dist_multi_gpu(
        const int inst, const int lb, const int D, const int w, const int commSize, const int optimum,
        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
        unsigned long long int *expTreeProc, unsigned long long int *expSolProc, unsigned long long int *nStealsProc,
        double *timeKernelCall, double *timeIdle, double *workloadProc);

#ifdef __cplusplus
}
#endif

#endif // PFSP_STATISTIC_H
