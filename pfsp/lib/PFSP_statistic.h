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
        const int inst, const int lb, const int D, int ws, const int optimum, const int m, const int M,
        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
        unsigned long long int *expTreeGPU, unsigned long long int *expSolGPU, unsigned long long int *genChildGPU,
        unsigned long long int *nbStealsGPU, unsigned long long int *nbSStealsGPU, unsigned long long int *nbTerminationGPU,
        double *timeGpuCpy, double *timeGpuMalloc, double *timeGpuKer, double *timeGenChild, double *timePoolOps,
        double *timeGpuIdle, double *timeTermination);

    void print_results_file_dist_multi_gpu(
        const int inst, const int lb, const int D, const int w, const int commSize, const int optimum, const int m, const int M,
        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
        unsigned long long int *all_expTreeGPU, unsigned long long int *all_expSolGPU, unsigned long long int *all_genChildGPU,
        unsigned long long int *all_nbStealsGPU, unsigned long long int *all_nbSStealsGPU, unsigned long long int *all_nbTerminationGPU,
        unsigned long long int *nbSDistLoadBal, double *all_timeGpuCpy, double *all_timeGpuMalloc, double *all_timeGpuKer,
        double *all_timeGenChild, double *all_timePoolOps, double *all_timeGpuIdle, double *all_timeTermination, double *timeLoadBal);

#ifdef __cplusplus
}
#endif

#endif // PFSP_STATISTIC_H
