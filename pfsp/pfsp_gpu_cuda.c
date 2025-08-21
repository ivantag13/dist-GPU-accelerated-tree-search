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
#include <sched.h>
#include <getopt.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"
#include "lib/PFSP_gpu_lib.cuh"
#include "lib/PFSP_lib.h"
#include "lib/Pool_atom.h"
#include "lib/PFSP_statistic.h"
#include "../common/util.h"
#include "../common/gpu_util.cuh"

/*******************************************************************************
Implementation of the parallel single-GPU PFSP search.
*******************************************************************************/
void pfsp_search(const int inst, const int lb, const int m, const int M, const int D, int *best, unsigned long long int *exploredTree,
                 unsigned long long int *exploredSol, double *elapsedTime, double *timeGpuCpy, double *timeGpuMalloc,
                 double *timeGpuKer, double *timeGenChild)
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
  struct timespec start, end, startGpuCpy, endGpuCpy, startGpuMalloc, endGpuMalloc, startGpuKer, endGpuKer, startGenChild, endGenChild;
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
  clock_gettime(CLOCK_MONOTONIC_RAW, &startGpuMalloc);

  // GPU bounding functions data
  lb1_bound_data lbound1_d;
  int *p_times_d, *min_heads_d, *min_tails_d;
  lb1_alloc_gpu(&lbound1_d, lbound1, p_times_d, min_heads_d, min_tails_d, jobs, machines);

  lb2_bound_data lbound2_d;
  int *johnson_schedule_d, *lags_d, *machine_pairs_1_d, *machine_pairs_2_d, *machine_pair_order_d;
  lb2_alloc_gpu(&lbound2_d, lbound2, johnson_schedule_d, lags_d, machine_pairs_1_d, machine_pairs_2_d, machine_pair_order_d, jobs, machines);

  // Allocating parents vector on CPU and GPU
  // TODO: rethink dynamic allocation of CPU memory?!
  Node *parents = (Node *)malloc(M * sizeof(Node));
  Node *children = (Node *)malloc(M * jobs * sizeof(Node));
  // Node parents[M], children[M * jobs];
  Node *parents_d;
  cudaMalloc((void **)&parents_d, M * sizeof(Node));

  int *sumOffSets = (int *)malloc(M * sizeof(int));
  // int sumOffSets[M];
  int *sumOffSets_d;
  cudaMalloc((void **)&sumOffSets_d, M * sizeof(int));

  // Allocating bounds vector on CPU and GPU
  int *nodeIndex = (int *)malloc((jobs * M) * sizeof(int));
  // int nodeIndex[jobs * M];
  int *nodeIndex_d;
  cudaMalloc((void **)&nodeIndex_d, (jobs * M) * sizeof(int));

  int *bounds = (int *)malloc((jobs * M) * sizeof(int));
  // int bounds[jobs * M];
  int *bounds_d;
  cudaMalloc((void **)&bounds_d, (jobs * M) * sizeof(int));

  clock_gettime(CLOCK_MONOTONIC_RAW, &endGpuMalloc);
  *timeGpuMalloc = (endGpuMalloc.tv_sec - startGpuMalloc.tv_sec) + (endGpuMalloc.tv_nsec - startGpuMalloc.tv_nsec) / 1e9;

  int counter = 1;
  while (1)
  {
    // Node parents[M];
    int poolSize = popBackBulkFree(&pool, m, M, parents, 1);

    if (poolSize > 0)
    {
      clock_gettime(CLOCK_MONOTONIC_RAW, &startGpuCpy);
      int sum = 0;
      int diff;
      int i, j;
      // int sumOffSets[poolSize];
      // int nodeIndex[poolSize * jobs];
      for (i = 0; i < poolSize; i++)
      {
        diff = jobs - parents[i].depth;
        for (j = 0; j < diff; j++)
          nodeIndex[j + sum] = i;
        sum += diff;
        sumOffSets[i] = sum;
      }
      const int numBounds = sum;
      const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);

      cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);
      cudaMemcpy(sumOffSets_d, sumOffSets, poolSize * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(nodeIndex_d, nodeIndex, numBounds * sizeof(int), cudaMemcpyHostToDevice);
      clock_gettime(CLOCK_MONOTONIC_RAW, &endGpuCpy);
      *timeGpuCpy += (endGpuCpy.tv_sec - startGpuCpy.tv_sec) + (endGpuCpy.tv_nsec - startGpuCpy.tv_nsec) / 1e9;

      // numBounds is the 'size' of the problem
      clock_gettime(CLOCK_MONOTONIC_RAW, &startGpuKer);
      evaluate_gpu(jobs, lb, numBounds, nbBlocks, poolSize, best, lbound1_d, lbound2_d, parents_d, bounds_d, sumOffSets_d, nodeIndex_d);
      cudaDeviceSynchronize();
      clock_gettime(CLOCK_MONOTONIC_RAW, &endGpuKer);
      *timeGpuKer += (endGpuKer.tv_sec - startGpuKer.tv_sec) + (endGpuKer.tv_nsec - startGpuKer.tv_nsec) / 1e9;

      clock_gettime(CLOCK_MONOTONIC_RAW, &startGpuCpy);
      // int bounds[numBounds];
      cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);
      clock_gettime(CLOCK_MONOTONIC_RAW, &endGpuCpy);
      *timeGpuCpy += (endGpuCpy.tv_sec - startGpuCpy.tv_sec) + (endGpuCpy.tv_nsec - startGpuCpy.tv_nsec) / 1e9;

      // each task generates and inserts its children nodes to the pool.
      // TODO: add OpenMP loop to parallelize multiple uses of generate_children
      clock_gettime(CLOCK_MONOTONIC_RAW, &startGenChild);

      int threadPoolSize[D], parentsStart[D], childrenStart[D];
      printf("\n pool.size[%d] poolSize[%d] threadPoolSize: ", pool.size, poolSize);
      for (int l = 0; l < D; l++)
      {
        // Amount of parents nodes to be treated per OpenMP thread
        threadPoolSize[l] = poolSize / D;
        if (l < (poolSize % D))
          threadPoolSize[l]++;

        parentsStart[l] = 0;
        for (int k = 0; k < l; k++)
          parentsStart[l] += threadPoolSize[k];

        childrenStart[l] = 0;
        int totalParents = 0;
        for (int k = 0; k < l; k++)
          totalParents += threadPoolSize[k];
        if (l > 0)
        {
          // totalParents--;
          childrenStart[l] = sumOffSets[totalParents];
        }
        printf("%d ", threadPoolSize[l]);
      }
      printf("\n");

      unsigned long long int localExploredTree = 0, localExploredSol = 0;
      int localBest = *best;
#pragma omp parallel num_threads(D) shared(parents, children, poolSize, jobs, bounds, exploredTree, exploredSol, best, pool, threadPoolSize, D) \
    reduction(+ : localExploredSol, localExploredTree) reduction(min : localBest)
      {
        int threadId = omp_get_thread_num();
        int indexChildren;
        // int num_procs = omp_get_num_procs();   // int cpu = sched_getcpu();   // printf("Thread %d of %d (D[%d]) sees %d processors, & cpu is %d\n", threadId, omp_get_num_threads(), D, num_procs, cpu);
        //  generate_children(parents, children, poolSize, jobs, bounds, exploredTree, exploredSol, best, &indexChildren);

        printf("Thread[%d] parentsStart[%d] threadPoolSize[%d] childrenStart[%d]\n", threadId, parentsStart[threadId], threadPoolSize[threadId], childrenStart[threadId]);
        generate_children(parents + parentsStart[threadId], children + childrenStart[threadId], threadPoolSize[threadId], jobs, bounds + childrenStart[threadId], &localExploredTree, &localExploredSol, &localBest, &indexChildren);
        printf("Thread[%d] indexChildren[%d]\n", threadId, indexChildren);

#pragma omp critical
        {
          pushBackBulk(&pool, children + childrenStart[threadId], indexChildren);
        }
        // pushBackBulkFree(&pool, children, indexChildren);
      } // End of OpenMP parallel region
      *exploredTree += localExploredTree;
      *exploredSol += localExploredSol;
      *best = MIN(*best, localBest);
      printf("\n");

      clock_gettime(CLOCK_MONOTONIC_RAW, &endGenChild);
      *timeGenChild += (endGenChild.tv_sec - startGenChild.tv_sec) + (endGenChild.tv_nsec - startGenChild.tv_nsec) / 1e9;

      if (counter % 5000 == 0)
      {
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        double t2 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("Counter[%d]: Tree[%llu] Sol[%llu]\n Pool: size[%d] capacity[%d] poolSize[%d]\n Timer: Total[%f] cudaMemcpy[%f] cudaMalloc[%f] kernelCall[%f] generateChildren[%f]\n",
               counter, *exploredTree, *exploredSol, pool.size, pool.capacity, poolSize, t2, *timeGpuCpy, *timeGpuMalloc, *timeGpuKer, *timeGenChild);
      }
      counter++;
    }
    else
      break;
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double t2 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("\nSearch on GPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t2);
  // double achievedGOPS = (double)totalFlops / (double)(*timeKernelCall * 1e9);
  // double AI = (double)totalFlops / (double)totalBytes;
  // printf("Achieved GFLOPS: %f\n", achievedGOPS);
  // printf("Arithmetic Intensity: %f\n", AI);

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

  // Free Memory
  deleteSinglePool_atom(&pool);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);
  cudaFree(parents_d);
  cudaFree(bounds_d);
  cudaFree(sumOffSets_d);
  cudaFree(nodeIndex_d);
  cudaFree(p_times_d);
  cudaFree(min_heads_d);
  cudaFree(min_tails_d);
  cudaFree(johnson_schedule_d);
  cudaFree(lags_d);
  cudaFree(machine_pairs_1_d);
  cudaFree(machine_pairs_2_d);
  cudaFree(machine_pair_order_d);
  free(parents);
  free(sumOffSets);
  free(nodeIndex);
  free(children);
  free(bounds);

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double t3 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  *elapsedTime = t1 + t2 + t3;
  printf("\nSearch on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t3);
  printf("Times: Total[%f] cudaMemcpy[%f] cudaMalloc[%f] kernelCall[%f] generateChildren[%f]\n", *elapsedTime, *timeGpuCpy, *timeGpuMalloc, *timeGpuKer, *timeGenChild);
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

  double elapsedTime = 0, timeGpuCpy = 0, timeGpuMalloc = 0, timeGpuKer = 0, timeGenChild = 0;

  pfsp_search(inst, lb, m, M, D, &optimum, &exploredTree, &exploredSol, &elapsedTime, &timeGpuCpy, &timeGpuMalloc, &timeGpuKer, &timeGenChild);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  print_results_file_single_gpu(inst, lb, optimum, m, M, exploredTree, exploredSol, elapsedTime, timeGpuCpy, timeGpuMalloc, timeGpuKer, timeGenChild);

  return 0;
}
