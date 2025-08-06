/*******************************************************************************
Implementation of PFSP Statistic Storage and Analysis.
*******************************************************************************/
#include "PFSP_statistic.h"

// Write array fields as JSON-style strings
// Utility macro to format arrays (unsigned long long)
#define PRINT_ULL_ARRAY(arr)           \
    fprintf(file, "\"[");              \
    for (int i = 0; i < D; ++i)        \
    {                                  \
        fprintf(file, "%llu", arr[i]); \
        if (i != D - 1)                \
            fprintf(file, ",");        \
    }                                  \
    fprintf(file, "]\",")

// Utility macro to format arrays (double)
#define PRINT_DOUBLE_ARRAY(arr)        \
    fprintf(file, "\"[");              \
    for (int i = 0; i < D; ++i)        \
    {                                  \
        fprintf(file, "%.4f", arr[i]); \
        if (i != D - 1)                \
            fprintf(file, ",");        \
    }                                  \
    fprintf(file, "]\",")

#define PRINT_ULL_ARRAY_DIST(arr)          \
    fprintf(file, "\"[");                  \
    for (int i = 0; i < D * commSize; ++i) \
    {                                      \
        fprintf(file, "%llu", arr[i]);     \
        if (i != D * commSize - 1)                    \
            fprintf(file, ",");            \
    }                                      \
    fprintf(file, "]\",")

// Utility macro to format arrays (double)
#define PRINT_DOUBLE_ARRAY_DIST(arr)       \
    fprintf(file, "\"[");                  \
    for (int i = 0; i < D * commSize; ++i) \
    {                                      \
        fprintf(file, "%.4f", arr[i]);     \
        if (i != D * commSize - 1)                    \
            fprintf(file, ",");            \
    }                                      \
    fprintf(file, "]\",")

void print_results_file_single_gpu(const int inst, const int lb, const int optimum, const int m, const int M,
                                   const unsigned long long int exploredTree, const unsigned long long int exploredSol,
                                   const double timer, double timeGpuCpy, double timeGpuMalloc, double timeGpuKer, double timeGenChild)
{
    FILE *file = fopen("singlegpu.csv", "a");

    // Optional: add CSV header only once (check if file is empty)
    static int header_written = 0;
    if (!header_written)
    {
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        if (size == 0)
        {
            fprintf(file, "instance_id,lower_bound,optimum,m,M,total_time,gpu_memcpy_time,gpu_malloc_time,gpu_kernel_time,gen_child_time,explored_tree,explored_sol\n");
        }
        header_written = 1;
    }

    // Write data
    fprintf(file,
            "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%llu,%llu\n",
            inst, lb, optimum, m, M,
            timer, timeGpuCpy, timeGpuMalloc, timeGpuKer, timeGenChild,
            exploredTree, exploredSol);

    fclose(file);
}

void print_results_file_multi_gpu(
    const int inst, const int lb, const int D, int ws, const int optimum, const int m, const int M,
    const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
    unsigned long long int *expTreeGPU, unsigned long long int *expSolGPU, unsigned long long int *genChildGPU,
    unsigned long long int *nbStealsGPU, unsigned long long int *nbSStealsGPU, unsigned long long int *nbTerminationGPU,
    double *timeGpuCpy, double *timeGpuMalloc, double *timeGpuKer, double *timeGenChild, double *timePoolOps,
    double *timeGpuIdle, double *timeTermination)
{
    FILE *file = fopen("multigpu.csv", "a");

    // Write header only once
    static int header_written = 0;
    if (!header_written)
    {
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        if (size == 0)
        {
            fprintf(file,
                    "instance_id,nb_device,lower_bound,work_stealing,optimum,m,M,total_time,total_tree,total_sol,"
                    "exp_tree_gpu,exp_sol_gpu,gen_child_gpu,steals_gpu,success_steals_gpu,termination_gpu,"
                    "gpu_memcpy_time,gpu_malloc_time,gpu_kernel_time,gpu_gen_child_time,pool_ops_time,gpu_idle_time,termination_time\n");
        }
        header_written = 1;
    }

    // Write scalar values
    fprintf(file, "%d,%d,%d,%d,%d,%d,%d,%.4f,%llu,%llu,",
            inst, D, lb, ws, optimum, m, M, timer, exploredTree, exploredSol);

    // Arrays
    PRINT_ULL_ARRAY(expTreeGPU);
    PRINT_ULL_ARRAY(expSolGPU);
    PRINT_ULL_ARRAY(genChildGPU);
    PRINT_ULL_ARRAY(nbStealsGPU);
    PRINT_ULL_ARRAY(nbSStealsGPU);
    PRINT_ULL_ARRAY(nbTerminationGPU);
    PRINT_DOUBLE_ARRAY(timeGpuCpy);
    PRINT_DOUBLE_ARRAY(timeGpuMalloc);
    PRINT_DOUBLE_ARRAY(timeGpuKer);
    PRINT_DOUBLE_ARRAY(timeGenChild);
    PRINT_DOUBLE_ARRAY(timePoolOps);
    PRINT_DOUBLE_ARRAY(timeGpuIdle);
    PRINT_DOUBLE_ARRAY(timeTermination);

    // Finalize the row
    fprintf(file, "\n");

    // Clean up
    fclose(file);
}

void print_results_file_dist_multi_gpu(
    const int inst, const int lb, const int D, const int w, const int commSize, const int optimum, const int m, const int M,
    const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
    unsigned long long int *all_expTreeGPU, unsigned long long int *all_expSolGPU, unsigned long long int *all_genChildGPU,
    unsigned long long int *all_nbStealsGPU, unsigned long long int *all_nbSStealsGPU, unsigned long long int *all_nbTerminationGPU,
    unsigned long long int *nbSDistLoadBal, double *all_timeGpuCpy, double *all_timeGpuMalloc, double *all_timeGpuKer,
    double *all_timeGenChild, double *all_timePoolOps, double *all_timeGpuIdle, double *all_timeTermination, double *timeLoadBal)
{
    FILE *file = fopen("dist_multigpu.csv", "a");

    // Write header if file is new
    static int header_written = 0;
    if (!header_written)
    {
        fseek(file, 0, SEEK_END);
        if (ftell(file) == 0)
        {
            fprintf(file,
                    "instance_id,nb_device,comm_size,lower_bound,load_balancing,optimum,m,M,total_time,total_tree,total_sol,"
                    "all_exp_tree_gpu,all_exp_sol_gpu,all_gen_child_gpu,all_steals_gpu,all_success_steals_gpu,all_termination_gpu,"
                    "all_gpu_memcpy_time,all_gpu_malloc_time,all_gpu_kernel_time,all_gpu_gen_child_time,all_pool_ops_time,all_gpu_idle_time,all_termination_time\n");
        }
        header_written = 1;
    }

    // Write scalars
    fprintf(file, "%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%llu,%llu,",
            inst, D, commSize, lb, w, optimum, m, M, timer, exploredTree, exploredSol);

    // Write array fields
    PRINT_ULL_ARRAY_DIST(all_expTreeGPU);
    PRINT_ULL_ARRAY_DIST(all_expSolGPU);
    PRINT_ULL_ARRAY_DIST(all_genChildGPU);
    PRINT_ULL_ARRAY_DIST(all_nbStealsGPU);
    PRINT_ULL_ARRAY_DIST(all_nbSStealsGPU);
    PRINT_ULL_ARRAY_DIST(all_nbTerminationGPU);
    //PRINT_ULL_ARRAY_DIST(nbSDistLoadBal);

    PRINT_DOUBLE_ARRAY_DIST(all_timeGpuCpy);
    PRINT_DOUBLE_ARRAY_DIST(all_timeGpuMalloc);
    PRINT_DOUBLE_ARRAY_DIST(all_timeGpuKer);
    PRINT_DOUBLE_ARRAY_DIST(all_timeGenChild);
    PRINT_DOUBLE_ARRAY_DIST(all_timePoolOps);
    PRINT_DOUBLE_ARRAY_DIST(all_timeGpuIdle);
    PRINT_DOUBLE_ARRAY_DIST(all_timeTermination);
    //PRINT_DOUBLE_ARRAY_DIST(timeLoadBal);

    // Finalize row
    fprintf(file, "\n");

    // Clean up
    fclose(file);
}
