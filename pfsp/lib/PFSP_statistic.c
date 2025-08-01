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

void print_results_file_single_gpu(const int inst, const int lb, const int optimum,
                                   const unsigned long long int exploredTree, const unsigned long long int exploredSol,
                                   const double timer, double timeCudaMemCpy, double timeCudaMalloc, double timeKernelCall)
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
            fprintf(file, "instance_id,lower_bound,optimum,total_time,memcpy_time,cuda_malloc_time,kernel_call_time,explored_tree,explored_sol\n");
        }
        header_written = 1;
    }

    // Write data
    fprintf(file,
            "%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%llu,%llu\n",
            inst, lb, optimum,
            timer, timeCudaMemCpy, timeCudaMalloc, timeKernelCall,
            exploredTree, exploredSol);

    fclose(file);
}

void print_results_file_multi_gpu(
    const int inst, const int lb, const int D, int ws, const int optimum,
    const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
    unsigned long long int *expTreeGPU, unsigned long long int *expSolGPU, unsigned long long int *genChildren,
    unsigned long long int *nStealsGPU, unsigned long long int *nSStealsGPU, unsigned long long int *nTerminationGPU,
    double *timeCudaMemCpy, double *timeCudaMalloc, double *timeKernelCall, double *timeIdle, double *timeTermination,
    double *timePoolOps, double *timeGenChildren)
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
                    "instance_id,nb_device,lower_bound,work_sharing,optimum,total_time,total_tree,total_sol,"
                    "exp_tree_gpu,exp_sol_gpu,gen_children_gpu,steals_gpu,successful_steals_gpu,termination_gpu,"
                    "kernel_time_gpu,memcpy_time_gpu,malloc_time_gpu,gen_children_time,idle_time_gpu,termination_time_gpu\n");
        }
        header_written = 1;
    }

    // Write scalar values
    fprintf(file, "%d,%d,%d,%d,%d,%.4f,%llu,%llu,",
            inst, D, lb, ws, optimum, timer, exploredTree, exploredSol);

    // Arrays
    PRINT_ULL_ARRAY(expTreeGPU);
    PRINT_ULL_ARRAY(expSolGPU);
    PRINT_ULL_ARRAY(genChildren);
    PRINT_ULL_ARRAY(nStealsGPU);
    PRINT_ULL_ARRAY(nSStealsGPU);
    PRINT_ULL_ARRAY(nTerminationGPU);
    PRINT_DOUBLE_ARRAY(timeKernelCall);
    PRINT_DOUBLE_ARRAY(timeCudaMemCpy);
    PRINT_DOUBLE_ARRAY(timeCudaMalloc);
    PRINT_DOUBLE_ARRAY(timeGenChildren);
    PRINT_DOUBLE_ARRAY(timeIdle);
    PRINT_DOUBLE_ARRAY(timeTermination);

    // Finalize the row
    fprintf(file, "\n");

    // Clean up
    fclose(file);
}

void print_results_file_dist_multi_gpu(
    const int inst, const int lb, const int D, const int w, const int commSize, const int optimum,
    const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
    unsigned long long int *expTreeProc, unsigned long long int *expSolProc, unsigned long long int *nStealsProc,
    double *timeKernelCall, double *timeIdle, double *workloadProc)
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
                    "instance_id,lower_bound,nb_devices,work_sharing,nb_procs,optimum,"
                    "total_time,total_tree,total_sol,"
                    "explored_tree_per_proc,explored_sol_per_proc,steals_per_proc,"
                    "kernel_time_per_proc,idle_time_per_proc,workload_per_proc\n");
        }
        header_written = 1;
    }

    // Write scalars
    fprintf(file, "%d,%d,%d,%d,%d,%d,%.4f,%llu,%llu,",
            inst, lb, D, w, commSize, optimum,
            timer, exploredTree, exploredSol);

    // Write array fields
    PRINT_ULL_ARRAY(expTreeProc);
    PRINT_ULL_ARRAY(expSolProc);
    PRINT_ULL_ARRAY(nStealsProc);
    PRINT_DOUBLE_ARRAY(timeKernelCall);
    PRINT_DOUBLE_ARRAY(timeIdle);
    PRINT_DOUBLE_ARRAY(workloadProc);

    // Finalize row
    fprintf(file, "\n");

    // Clean up
    fclose(file);
}
