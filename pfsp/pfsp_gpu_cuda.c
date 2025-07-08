/*
  Single-GPU B&B to solve Taillard instances of the PFSP in C+CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <getopt.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"
#include "lib/PFSP_gpu_lib.cuh"
#include "lib/PFSP_lib.h"
#include "lib/Pool_atom.h"
#include "../common/util.h"
#include "../common/gpu_util.cuh"

/*******************************************************************************
FLOP estimation
*******************************************************************************/
// TODO: Fix FLOP estimations or use appropriate tool to measure it
// Number of machine‑pairs as a function of M
//   P = M*(M‑1)/2
static inline int P_of(int M)
{
  return (int)M * (M - 1) / 2;
}

// FLOP count per single lb1 bound invocation,
// as a function of N (jobs), M (machines), and lim (limit1):
static inline int flop_lb1(int N, int M, int lim)
{
  // 1) schedule_front: (lim+1) * [1 add + 2*(M‑1) ops]
  int F_front = (int)(lim) * (1 + 2 * (M - 1));
  // 2) sum_unscheduled: (N‑(lim+1)) * M adds
  int F_remain = (int)(N - (lim)) * M;
  // 3) machine_bound_from_parts: 2 + 4*(M‑1)
  int F_bind = 2 + 4 * (M - 1);
  return F_front + F_remain + F_bind;
}

// Bytes per single lb1‑bound invocation:
//  - loads  = N*M ints  (schedule_front + sum_unscheduled)
//  - stores = 1 int      (write one bounds entry)
//  - total bytes = 4 bytes/Int * (loads + stores)
static inline int bytes_per_inv_lb1(int N, int M)
{
  int loads = (int)N * M;
  int stores = 1;
  return 4 * (loads + stores);
}

// FLOP count per single lb2 bound invocation:
//   same front/remain cost as lb1, plus the lb_makespan loop over P pairs
static inline int flop_lb2(int N, int M, int lim)
{
  int F_front = (int)(lim) * (1 + 2 * (M - 1));
  int F_remain = (int)(N - (lim)) * M;
  int P = P_of(M);
  // For each pair: 4 ops per unscheduled job + 4 final ops
  int unsched = (int)(N - (lim));
  int F_pair = 4 * unsched + 4;
  int F_makesp = P * F_pair;
  return F_front + F_remain + F_makesp;
}

// Bytes per single lb2‑bound invocation:
//  - loads  = N*M                           (front + remain)
//           + 3 * P * (N - lim - 1)         (2 p_times + 1 lag per unscheduled job per pair)
//  - stores = 1
static inline int bytes_per_inv_lb2(int N, int M, int lim)
{
  int P = P_of(M);
  int uns = (int)N - lim;
  int loads = (int)N * M + 3 * P * uns;
  int stores = 1;
  return 4 * (loads + stores);
}

/*******************************************************************************
Statistics Functions
*******************************************************************************/
void print_results_file(const int inst, const int machines, const int jobs, const int lb, const int optimum,
                        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer, double timeCudaMemCpy, double timeCudaMalloc, double timeKernelCall)
{
  FILE *file;
  file = fopen("gpu.dat", "a");
  fprintf(file, "S-GPU-opt ta%d lb%d Time[%.4f] memCpy[%.4f] cudaMalloc[%.4f] kernelCall[%.4f] Tree[%llu] Sol[%llu] Best[%d]\n", inst, lb, timer, timeCudaMemCpy, timeCudaMalloc, timeKernelCall, exploredTree, exploredSol, optimum);
  fclose(file);
  return;
}

/*******************************************************************************
Implementation of the parallel CUDA GPU PFSP search.
*******************************************************************************/
void pfsp_search(const int inst, const int lb, const int m, const int M, int *best, unsigned long long int *exploredTree,
                 unsigned long long int *exploredSol, double *elapsedTime, double *timeCudaMemCpy, double *timeCudaMalloc, double *timeKernelCall)
{
  gpu_info();

  // Initializing problem
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  // Starting pool
  Node root;
  initRoot(&root, jobs);

  SinglePool_atom pool;
  initSinglePool_atom(&pool);

  pushBack(&pool, root);

  // Timers
  struct timespec start, end, startCudaMemCpy, endCudaMemCpy, startCudaMalloc, endCudaMalloc, startKernelCall, endKernelCall, startGenChildren, endGenChildren;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  // Bounding data
  lb1_bound_data *lbound1;
  lbound1 = new_bound_data(jobs, machines);
  taillard_get_processing_times(lbound1->p_times, inst);
  fill_min_heads_tails(lbound1);

  lb2_bound_data *lbound2;
  lbound2 = new_johnson_bd_data(lbound1);
  fill_machine_pairs(lbound2 /*, LB2_FULL*/);
  fill_lags(lbound1->p_times, lbound2);
  fill_johnson_schedules(lbound1->p_times, lbound2);

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */

  while (pool.size < m)
  {
    // CPU side
    int hasWork = 0;
    Node parent = popFrontFree(&pool, &hasWork);
    if (!hasWork)
      break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double t1 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nInitial search on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t1);

  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  clock_gettime(CLOCK_MONOTONIC_RAW, &startCudaMalloc);

  // GPU bounding functions data
  lb1_bound_data lbound1_d;
  int *p_times_d, *min_heads_d, *min_tails_d;
  lb1_alloc_gpu(&lbound1_d, lbound1, p_times_d, min_heads_d, min_tails_d, jobs, machines);

  lb2_bound_data lbound2_d;
  int *johnson_schedule_d, *lags_d, *machine_pairs_1_d, *machine_pairs_2_d, *machine_pair_order_d;
  lb2_alloc_gpu(&lbound2_d, lbound2, johnson_schedule_d, lags_d, machine_pairs_1_d, machine_pairs_2_d, machine_pair_order_d, jobs, machines);

  // Allocating parents vector on CPU and GPU
  Node *parents = (Node *)malloc(M * sizeof(Node));
  Node *children = (Node *)malloc(M * jobs * sizeof(Node));
  Node *parents_d;
  cudaMalloc((void **)&parents_d, M * sizeof(Node));

  int *sumOffSets = (int *)malloc(M * sizeof(int));
  int *sumOffSets_d;
  cudaMalloc((void **)&sumOffSets_d, M * sizeof(int));

  // Allocating bounds vector on CPU and GPU
  int *nodeIndex = (int *)malloc((jobs * M) * sizeof(int));
  int *nodeIndex_d;
  cudaMalloc((void **)&nodeIndex_d, (jobs * M) * sizeof(int));

  int *bounds = (int *)malloc((jobs * M) * sizeof(int));
  int *bounds_d;
  cudaMalloc((void **)&bounds_d, (jobs * M) * sizeof(int));

  clock_gettime(CLOCK_MONOTONIC_RAW, &endCudaMalloc);

  *timeCudaMalloc = (endCudaMalloc.tv_sec - startCudaMalloc.tv_sec) + (endCudaMalloc.tv_nsec - startCudaMalloc.tv_nsec) / 1e9;

  // int counter = 0;
  int totalFlops = 0;
  int totalBytes = 0;
  int indexChildren;
  double timeGenChildren = 0;

  while (1)
  {
    // int poolSize = pool.size;
    // TODO : fix call of popBackBulkFree to use it here
    int poolSize = popBackBulk(&pool, m, M, parents);

    if (poolSize > 0)
    {
      clock_gettime(CLOCK_MONOTONIC_RAW, &startCudaMemCpy);
      int sum = 0;
      int diff;
      int i, j;
      int lim;
      for (i = 0; i < poolSize; i++)
      {
        lim = parents[i].limit1 + 1;
        diff = jobs - parents[i].depth;
        for (j = 0; j < diff; j++)
          nodeIndex[j + sum] = i;
        sum += diff;
        sumOffSets[i] = sum;

        if (jobs - lim < 0)
          printf("ERROR\n");
        int F = (lb == 1) ? flop_lb1(jobs, machines, lim) : flop_lb2(jobs, machines, lim);
        int per_inv = (lb == 1) ? bytes_per_inv_lb1(jobs, machines) : bytes_per_inv_lb2(jobs, machines, lim);
        // each parent node issues (N - lim) bound calls:
        totalFlops += (jobs - lim) * F;
        totalBytes += (jobs - lim) * per_inv;
      }
      const int numBounds = sum;
      const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);

      cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);
      cudaMemcpy(sumOffSets_d, sumOffSets, poolSize * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(nodeIndex_d, nodeIndex, numBounds * sizeof(int), cudaMemcpyHostToDevice);
      clock_gettime(CLOCK_MONOTONIC_RAW, &endCudaMemCpy);
      *timeCudaMemCpy += (endCudaMemCpy.tv_sec - startCudaMemCpy.tv_sec) + (endCudaMemCpy.tv_nsec - startCudaMemCpy.tv_nsec) / 1e9;

      // numBounds is the 'size' of the problem
      clock_gettime(CLOCK_MONOTONIC_RAW, &startKernelCall);
      evaluate_gpu(jobs, lb, numBounds, nbBlocks, poolSize, best, lbound1_d, lbound2_d, parents_d, bounds_d, sumOffSets_d, nodeIndex_d);
      cudaDeviceSynchronize();
      clock_gettime(CLOCK_MONOTONIC_RAW, &endKernelCall);
      *timeKernelCall += (endKernelCall.tv_sec - startKernelCall.tv_sec) + (endKernelCall.tv_nsec - startKernelCall.tv_nsec) / 1e9;

      clock_gettime(CLOCK_MONOTONIC_RAW, &startCudaMemCpy);
      cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);
      clock_gettime(CLOCK_MONOTONIC_RAW, &endCudaMemCpy);
      *timeCudaMemCpy += (endCudaMemCpy.tv_sec - startCudaMemCpy.tv_sec) + (endCudaMemCpy.tv_nsec - startCudaMemCpy.tv_nsec) / 1e9;

      /*
        each task generates and inserts its children nodes to the pool.
      */
      clock_gettime(CLOCK_MONOTONIC_RAW, &startGenChildren);
      generate_children(parents, children, poolSize, jobs, bounds, exploredTree, exploredSol, best, &pool, &indexChildren);
      pushBackBulkFree(&pool, children, indexChildren);
      clock_gettime(CLOCK_MONOTONIC_RAW, &endGenChildren);
      timeGenChildren += (endGenChildren.tv_sec - startGenChildren.tv_sec) + (endGenChildren.tv_nsec - startGenChildren.tv_nsec) / 1e9;

      // counter++;
    }
    else
    {
      break;
    }
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double t2 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  double achievedGOPS = (double)totalFlops / (double)(*timeKernelCall * 1e9);
  double AI = (double)totalFlops / (double)totalBytes;

  printf("\nSearch on GPU completed pool.capacity = %d\n", pool.capacity);
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t2);
  printf("Achieved GFLOPS: %f\n", achievedGOPS);
  printf("Arithmetic Intensity: %f\n", AI);
  printf("Time in Generate Children: %f\n", timeGenChildren);

  /*
    Step 3: We complete the depth-first search on CPU.
  */

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  while (1)
  {
    int hasWork = 0;
    Node parent = popBackFree(&pool, &hasWork);
    if (!hasWork)
      break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }

  // Freeing memory for structs
  deleteSinglePool_atom(&pool);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);

  // Freeing memory for device
  cudaFree(parents_d);
  cudaFree(bounds_d);
  cudaFree(p_times_d);
  cudaFree(min_heads_d);
  cudaFree(min_tails_d);
  cudaFree(johnson_schedule_d);
  cudaFree(lags_d);
  cudaFree(machine_pairs_1_d);
  cudaFree(machine_pairs_2_d);
  cudaFree(machine_pair_order_d);

  // Freeing memory for host
  free(parents);
  free(bounds);

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double t3 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  *elapsedTime = t1 + t2 + t3;
  printf("\nSearch on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t3);
  printf("Times: Total[%f] cudaMemcpy[%f] cudaMalloc[%f] kernelCall[%f]\n", *elapsedTime, *timeCudaMemCpy, *timeCudaMalloc, *timeKernelCall);
  printf("\nExploration terminated.\n");
}

int main(int argc, char *argv[])
{
  int version = 1; // Sequential version is code 0
  // Single-GPU PFSP only uses: inst, lb, ub, m, M
  int inst, lb, ub, m, M, D, ws, LB, commSize = 1; // commSize is an artificial variable here
  double perc;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &D, &ws, &LB, &perc);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  print_settings(inst, machines, jobs, ub, lb, D, ws, commSize, LB, version);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime, timeCudaMemCpy = 0, timeCudaMalloc = 0, timeKernelCall = 0;

  pfsp_search(inst, lb, m, M, &optimum, &exploredTree, &exploredSol, &elapsedTime, &timeCudaMemCpy, &timeCudaMalloc, &timeKernelCall);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  print_results_file(inst, machines, jobs, lb, optimum, exploredTree, exploredSol, elapsedTime, timeCudaMemCpy, timeCudaMalloc, timeKernelCall);

  return 0;
}
