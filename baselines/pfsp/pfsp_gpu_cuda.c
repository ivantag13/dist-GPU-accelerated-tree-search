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
#include "lib/evaluate.h"
#include "lib/Pool.h"

/******************************************************************************
CUDA functions
******************************************************************************/

#define gpuErrchk(ans)                          \
  {                                             \
    gpuAssert((ans), __FILE__, __LINE__, true); \
  }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

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
Implementation of the parallel CUDA GPU PFSP search.
*******************************************************************************/

void parse_parameters(int argc, char *argv[], int *inst, int *lb, int *ub, int *m, int *M)
{
  *m = 25;
  *M = 50000;
  *inst = 14;
  *lb = 1;
  *ub = 1;
  /*
    NOTE: Only forward branching is considered because other strategies increase a
    lot the implementation complexity and do not add much contribution.
  */

  // Define long options
  static struct option long_options[] = {
      {"inst", required_argument, NULL, 'i'},
      {"lb", required_argument, NULL, 'l'},
      {"ub", required_argument, NULL, 'u'},
      {"m", required_argument, NULL, 'm'},
      {"M", required_argument, NULL, 'M'},
      {NULL, 0, NULL, 0} // Terminate options array
  };

  int opt, value;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "i:l:u:m:M:", long_options, &option_index)) != -1)
  {
    value = atoi(optarg);

    switch (opt)
    {
    case 'i':
      if (value < 1 || value > 120)
      {
        fprintf(stderr, "Error: unsupported Taillard's instance\n");
        exit(EXIT_FAILURE);
      }
      *inst = value;
      break;

    case 'l':
      if (value < 0 || value > 2)
      {
        fprintf(stderr, "Error: unsupported lower bound function\n");
        exit(EXIT_FAILURE);
      }
      *lb = value;
      break;

    case 'u':
      if (value != 0 && value != 1)
      {
        fprintf(stderr, "Error: unsupported upper bound initialization\n");
        exit(EXIT_FAILURE);
      }
      *ub = value;
      break;

    case 'm':
      if (value < 1)
      {
        fprintf(stderr, "Error: unsupported minimal pool for GPU initialization\n");
        exit(EXIT_FAILURE);
      }
      *m = value;
      break;

    case 'M':
      if (value < *m)
      {
        fprintf(stderr, "Error: unsupported maximal pool for GPU initialization\n");
        exit(EXIT_FAILURE);
      }
      *M = value;
      break;

    default:
      fprintf(stderr, "Usage: %s --inst <value> --lb <value> --ub <value> --m <value> --M <value>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb)
{
  printf("\n=================================================\n");
  printf("Single-GPU C+CUDA\n\n");
  printf("Resolution of PFSP Taillard's instance: ta%d (m = %d, n = %d)\n", inst, machines, jobs);
  if (ub == 0)
    printf("Initial upper bound: inf\n");
  else /* if (ub == 1) */
    printf("Initial upper bound: opt\n");
  if (lb == 0)
    printf("Lower bound function: lb1_d\n");
  else if (lb == 1)
    printf("Lower bound function: lb1\n");
  else /* (lb == 2) */
    printf("Lower bound function: lb2\n");
  printf("Branching rule: fwd\n");
  printf("=================================================\n");
}

void print_results(const int optimum, const unsigned long long int exploredTree,
                   const unsigned long long int exploredSol, const double timer)
{
  printf("\n=================================================\n");
  printf("Size of the explored tree: %llu\n", exploredTree);
  printf("Number of explored solutions: %llu\n", exploredSol);
  /* TODO: Add 'is_better' */
  printf("Optimal makespan: %d\n", optimum);
  printf("Elapsed time: %.4f [s]\n", timer);
  printf("=================================================\n");
}

void print_results_file(const int inst, const int machines, const int jobs, const int lb, const int optimum,
                        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer, double timeCudaMemCpy, double timeCudaMalloc, double timeKernelCall)
{
  FILE *file;
  file = fopen("stats_pfsp_gpu_cuda.dat", "a");
  fprintf(file, "S-GPU ta%d lb%d Time[%.4f] memCpy[%.4f] cudaMalloc[%.4f] kernelCall[%.4f] Tree[%llu] Sol[%llu] Best[%d]\n", inst, lb, timer, timeCudaMemCpy, timeCudaMalloc, timeKernelCall, exploredTree, exploredSol, optimum);
  fclose(file);
  return;
}

inline void swap(int *a, int *b)
{
  int tmp = *b;
  *b = *a;
  *a = tmp;
}

// Evaluate and generate children nodes on CPU.
void decompose_lb1(const int jobs, const lb1_bound_data *const lbound1, const Node parent,
                   int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool *pool)
{
  for (int i = parent.limit1 + 1; i < jobs; i++)
  {
    Node child;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
    swap(&child.prmu[parent.depth], &child.prmu[i]);
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;

    int lowerbound = lb1_bound(lbound1, child.prmu, child.limit1, jobs);

    if (child.depth == jobs)
    { // if child leaf
      *num_sol += 1;

      if (lowerbound < *best)
      { // if child feasible
        *best = lowerbound;
      }
    }
    else
    { // if not leaf
      if (lowerbound < *best)
      { // if child feasible
        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }
}

void decompose_lb1_d(const int jobs, const lb1_bound_data *const lbound1, const Node parent,
                     int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool *pool)
{
  int *lb_begin = (int *)malloc(jobs * sizeof(int));

  lb1_children_bounds(lbound1, parent.prmu, parent.limit1, jobs, lb_begin);

  for (int i = parent.limit1 + 1; i < jobs; i++)
  {
    const int job = parent.prmu[i];
    const int lb = lb_begin[job];

    if (parent.depth + 1 == jobs)
    { // if child leaf
      *num_sol += 1;

      if (lb < *best)
      { // if child feasible
        *best = lb;
      }
    }
    else
    { // if not leaf
      if (lb < *best)
      { // if child feasible
        Node child;
        memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
        child.depth = parent.depth + 1;
        child.limit1 = parent.limit1 + 1;
        swap(&child.prmu[child.limit1], &child.prmu[i]);

        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }

  free(lb_begin);
}

void decompose_lb2(const int jobs, const lb1_bound_data *const lbound1, const lb2_bound_data *const lbound2,
                   const Node parent, int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol,
                   SinglePool *pool)
{
  for (int i = parent.limit1 + 1; i < jobs; i++)
  {
    Node child;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
    swap(&child.prmu[parent.depth], &child.prmu[i]);
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;

    int lowerbound = lb2_bound(lbound1, lbound2, child.prmu, child.limit1, jobs, *best);

    if (child.depth == jobs)
    { // if child leaf
      *num_sol += 1;

      if (lowerbound < *best)
      { // if child feasible
        *best = lowerbound;
      }
    }
    else
    { // if not leaf
      if (lowerbound < *best)
      { // if child feasible
        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }
}

void decompose(const int jobs, const int lb, int *best, const lb1_bound_data *const lbound1,
               const lb2_bound_data *const lbound2, const Node parent, unsigned long long int *tree_loc,
               unsigned long long int *num_sol, SinglePool *pool)
{
  switch (lb)
  {
  case 0: // lb1_d
    decompose_lb1_d(jobs, lbound1, parent, best, tree_loc, num_sol, pool);
    break;

  case 1: // lb1
    decompose_lb1(jobs, lbound1, parent, best, tree_loc, num_sol, pool);
    break;

  case 2: // lb2
    decompose_lb2(jobs, lbound1, lbound2, parent, best, tree_loc, num_sol, pool);
    break;
  }
}

// Generate children nodes (evaluated on GPU) on CPU
void generate_children(Node *parents, const int size, const int jobs, int *bounds,
                       unsigned long long int *exploredTree, unsigned long long int *exploredSol, int *best, SinglePool *pool)
{
  for (int i = 0; i < size; i++)
  {
    Node parent = parents[i];
    const uint8_t depth = parent.depth;

    for (int j = parent.limit1 + 1; j < jobs; j++)
    {
      const int lowerbound = bounds[j + i * jobs];

      // If child leaf
      if (depth + 1 == jobs)
      {
        *exploredSol += 1;

        // If child feasible
        if (lowerbound < *best)
          *best = lowerbound;
      }
      else
      { // If not leaf
        if (lowerbound < *best)
        {
          Node child;
          memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
          swap(&child.prmu[depth], &child.prmu[j]);
          child.depth = depth + 1;
          child.limit1 = parent.limit1 + 1;

          pushBack(pool, child);
          *exploredTree += 1;
        }
      }
    }
  }
}

// Single-GPU PFSP search
void pfsp_search(const int inst, const int lb, const int m, const int M, int *best, unsigned long long int *exploredTree,
                 unsigned long long int *exploredSol, double *elapsedTime, double *timeCudaMemCpy, double *timeCudaMalloc, double *timeKernelCall)
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

  // Initializing problem
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  // Starting pool
  Node root;
  initRoot(&root, jobs);

  SinglePool pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  // Timer
  struct timespec start, end, startCudaMemCpy, endCudaMemCpy, startCudaMalloc, endCudaMalloc, startKernelCall, endKernelCall;
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
    Node parent = popFront(&pool, &hasWork);
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

  // TODO: add function 'copyBoundsDevice' to perform the deep copy of bounding data
  // Vectors for deep copy of lbound1 to device
  lb1_bound_data lbound1_d;
  int *p_times_d;
  int *min_heads_d;
  int *min_tails_d;

  // Allocating and copying memory necessary for deep copy of lbound1
  cudaMalloc((void **)&p_times_d, jobs * machines * sizeof(int));
  cudaMalloc((void **)&min_heads_d, machines * sizeof(int));
  cudaMalloc((void **)&min_tails_d, machines * sizeof(int));
  cudaMemcpy(p_times_d, lbound1->p_times, (jobs * machines) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(min_heads_d, lbound1->min_heads, machines * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(min_tails_d, lbound1->min_tails, machines * sizeof(int), cudaMemcpyHostToDevice);

  // Deep copy of lbound1
  lbound1_d.p_times = p_times_d;
  lbound1_d.min_heads = min_heads_d;
  lbound1_d.min_tails = min_tails_d;
  lbound1_d.nb_jobs = lbound1->nb_jobs;
  lbound1_d.nb_machines = lbound1->nb_machines;

  // Vectors for deep copy of lbound2 to device
  lb2_bound_data lbound2_d;
  int *johnson_schedule_d;
  int *lags_d;
  int *machine_pairs_1_d;
  int *machine_pairs_2_d;
  int *machine_pair_order_d;

  // Allocating and copying memory necessary for deep copy of lbound2
  int nb_mac_pairs = lbound2->nb_machine_pairs;
  cudaMalloc((void **)&johnson_schedule_d, (nb_mac_pairs * jobs) * sizeof(int));
  cudaMalloc((void **)&lags_d, (nb_mac_pairs * jobs) * sizeof(int));
  cudaMalloc((void **)&machine_pairs_1_d, nb_mac_pairs * sizeof(int));
  cudaMalloc((void **)&machine_pairs_2_d, nb_mac_pairs * sizeof(int));
  cudaMalloc((void **)&machine_pair_order_d, nb_mac_pairs * sizeof(int));
  cudaMemcpy(johnson_schedule_d, lbound2->johnson_schedules, (nb_mac_pairs * jobs) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(lags_d, lbound2->lags, (nb_mac_pairs * jobs) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(machine_pairs_1_d, lbound2->machine_pairs_1, nb_mac_pairs * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(machine_pairs_2_d, lbound2->machine_pairs_2, nb_mac_pairs * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(machine_pair_order_d, lbound2->machine_pair_order, nb_mac_pairs * sizeof(int), cudaMemcpyHostToDevice);

  // Deep copy of lbound2
  lbound2_d.johnson_schedules = johnson_schedule_d;
  lbound2_d.lags = lags_d;
  lbound2_d.machine_pairs_1 = machine_pairs_1_d;
  lbound2_d.machine_pairs_2 = machine_pairs_2_d;
  lbound2_d.machine_pair_order = machine_pair_order_d;
  lbound2_d.nb_machine_pairs = lbound2->nb_machine_pairs;
  lbound2_d.nb_jobs = lbound2->nb_jobs;
  lbound2_d.nb_machines = lbound2->nb_machines;

  // Allocating parents vector on CPU and GPU
  Node *parents = (Node *)malloc(M * sizeof(Node));
  Node *parents_d;
  cudaMalloc((void **)&parents_d, M * sizeof(Node));

  // Allocating bounds vector on CPU and GPU
  int *bounds = (int *)malloc((jobs * M) * sizeof(int));
  int *bounds_d;
  cudaMalloc((void **)&bounds_d, (jobs * M) * sizeof(int));

  clock_gettime(CLOCK_MONOTONIC_RAW, &endCudaMalloc);

  *timeCudaMalloc = (endCudaMalloc.tv_sec - startCudaMalloc.tv_sec) + (endCudaMalloc.tv_nsec - startCudaMalloc.tv_nsec) / 1e9;

  int totalFlops = 0;
  int totalBytes = 0;

  while (1)
  {
    int poolSize = pool.size;
    if (poolSize >= m)
    {
      poolSize = MIN(poolSize, M);

      for (int i = 0; i < poolSize; i++)
      {
        int hasWork = 0;
        parents[i] = popBack(&pool, &hasWork);
        if (!hasWork)
          break;
      }

      /*
        TODO: Optimize 'numBounds' based on the fact that the maximum number of
        generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
        something like that.
      */

      for (int i = 0; i < poolSize; ++i)
      {
        int lim = parents[i].limit1+1;
        if(jobs-lim<0)
          printf("ERROR\n");
        int F = (lb == 1)
                    ? flop_lb1(jobs, machines, lim)
                    : flop_lb2(jobs, machines, lim);
        totalFlops += (jobs - lim) * F;
        int per_inv = (lb == 1)
                          ? bytes_per_inv_lb1(jobs, machines)
                          : bytes_per_inv_lb2(jobs, machines, lim);
        // each parent node issues (N - lim) bound calls:
        totalBytes += (jobs - lim) * per_inv;
      }

      const int numBounds = jobs * poolSize;
      const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);
      clock_gettime(CLOCK_MONOTONIC_RAW, &startCudaMemCpy);
      cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);
      clock_gettime(CLOCK_MONOTONIC_RAW, &endCudaMemCpy);
      *timeCudaMemCpy += (endCudaMemCpy.tv_sec - startCudaMemCpy.tv_sec) + (endCudaMemCpy.tv_nsec - startCudaMemCpy.tv_nsec) / 1e9;

      clock_gettime(CLOCK_MONOTONIC_RAW, &startKernelCall);
      // numBounds is the 'size' of the problem
      evaluate_gpu(jobs, lb, numBounds, nbBlocks, best, lbound1_d, lbound2_d, parents_d, bounds_d);
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
      generate_children(parents, poolSize, jobs, bounds, exploredTree, exploredSol, best, &pool);
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

  printf("\nSearch on GPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t2);
  printf("Achieved GFLOPS: %f\n", achievedGOPS);
  printf("Arithmetic Intensity: %f\n", AI);

  /*
    Step 3: We complete the depth-first search on CPU.
  */

  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  while (1)
  {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork)
      break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }

  // Freeing memory for structs
  deleteSinglePool(&pool);
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

  printf("Times: Total[%f] cudaMemCpy[%f] cudaMalloc[%f] kernelCall[%f]\n", *elapsedTime, *timeCudaMemCpy, *timeCudaMalloc, *timeKernelCall);

  printf("\nExploration terminated.\n");
}

int main(int argc, char *argv[])
{
  int inst, lb, ub, m, M;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  print_settings(inst, machines, jobs, ub, lb);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime, timeCudaMemCpy = 0, timeCudaMalloc = 0, timeKernelCall = 0;

  pfsp_search(inst, lb, m, M, &optimum, &exploredTree, &exploredSol, &elapsedTime, &timeCudaMemCpy, &timeCudaMalloc, &timeKernelCall);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  print_results_file(inst, machines, jobs, lb, optimum, exploredTree, exploredSol, elapsedTime, timeCudaMemCpy, timeCudaMalloc, timeKernelCall);

  return 0;
}
