/*
  Multi-core Multi-GPU Parallel CPU B&B to solve Taillard instances of the PFSP in C+OpenMP+CUDA.
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
#include <sched.h>
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
// #include "../common/gpu_util.cuh"

void checkBest(int *best_l, int *best, _Atomic bool *bestLock)
{
  if (*best_l > *best)
    *best_l = *best;
  else if (*best_l < *best)
  {
    bool expected = false;
    while (1)
    {
      expected = false;
      if (atomic_compare_exchange_strong(bestLock, &expected, true))
      {
        if (*best_l < *best)
          *best = *best_l;
        atomic_store(bestLock, false);
        break;
      }
    }
  }
  return;
}

/*******************************************************************************
Implementation of the parallel CPU PFSP search.
*******************************************************************************/
void pfsp_search(const int inst, const int lb, const int m, const int M, const int D, const double perc, int ws, int *best,
                 unsigned long long int *exploredTree, unsigned long long int *exploredSol, double *elapsedTime, unsigned long long int *expTreeGPU,
                 unsigned long long int *expSolGPU, unsigned long long int *genChildGPU, unsigned long long int *nbStealsGPU, unsigned long long int *nbSStealsGPU,
                 unsigned long long int *nbTerminationGPU, double *timeGpuCpy, double *timeGpuMalloc, double *timeGpuKer, double *timeGenChild,
                 double *timePoolOps, double *timeGpuIdle, double *timeTermination)
{
  int nb_procs = omp_get_num_procs();
  int MAX_GPU = 8;
  int NB_THREADS_GPU = (nb_procs / MAX_GPU);
  int NB_THREADS_MAX = D * NB_THREADS_GPU;

  printf("Num_procs[%d] MAX_GPU[%d] NB_THREADS_GPU[%d] NB_THREADS_MAX[%d]\n", nb_procs, MAX_GPU, NB_THREADS_GPU, NB_THREADS_MAX);

  // Initializing problem
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  // Starting pool
  Node root;
  initRoot(&root, jobs);

  SinglePool_atom pool;
  initSinglePool_atom(&pool);

  pushBack(&pool, root);

  // Boolean variables for termination detection
  _Atomic bool bestLock = false;
  _Atomic bool allTasksIdleFlag = false;
  _Atomic bool eachTaskState[NB_THREADS_MAX]; // one task per GPU
  for (int i = 0; i < NB_THREADS_MAX; i++)
    atomic_store(&eachTaskState[i], BUSY);

  // Timer
  double startTime, endTime;
  startTime = omp_get_wtime();

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

  while (pool.size < NB_THREADS_MAX * m)
  {
    int hasWork = 0;
    Node parent = popFrontFree(&pool, &hasWork);
    if (!hasWork)
      break;
    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }
  endTime = omp_get_wtime();
  double t1 = endTime - startTime;

  printf("\nInitial search on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t1);

  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */

  unsigned long long int eachExpTree[NB_THREADS_MAX], eachExpSol[NB_THREADS_MAX];
  SinglePool_atom multiPool[NB_THREADS_MAX];
  for (int i = 0; i < NB_THREADS_MAX; i++)
  {
    initSinglePool_atom(&multiPool[i]);
    eachExpTree[i] = 0;
    eachExpSol[i] = 0;
  }
  int best_l = *best;

#pragma omp parallel num_threads(NB_THREADS_MAX) shared(bestLock, eachTaskState, allTasksIdleFlag, pool, multiPool,                                         \
                                                            jobs, machines, lbound1, lbound2, lb, m, M, D, perc, ws, best, exploredTree, exploredSol,       \
                                                            elapsedTime, expTreeGPU, expSolGPU, genChildGPU, nbStealsGPU, nbSStealsGPU, nbTerminationGPU,   \
                                                            timeGpuCpy, timeGpuMalloc, timeGpuKer, timeGenChild, timePoolOps, timeGpuIdle, timeTermination) \
    reduction(min : best_l)
  {
    double startGpuCpy, endGpuCpy, startGpuKer, endGpuKer, startGenChild, endGenChild,
        startPoolOps, endPoolOps, startGpuIdle, endGpuIdle, startTermination, endTermination;
    int cpuID = omp_get_thread_num();
    int cpulb = lb;
    if (lb == 1)
      cpulb = 0;

    if (cpuID % NB_THREADS_GPU == 0)
      cudaSetDevice(cpuID / NB_THREADS_GPU);

    unsigned long long int tree = 0, sol = 0;
    int nbSteals = 0, nbSSteals = 0;
    SinglePool_atom *pool_loc;
    pool_loc = &multiPool[cpuID];
    SinglePool_atom parentsPool, childrenPool;
    initSinglePool_atom(&parentsPool);
    initSinglePool_atom(&childrenPool);

    // int best_l = *best;
    bool taskState = BUSY;

    roundRobin_distribution(pool_loc, &pool, cpuID, NB_THREADS_MAX);
#pragma omp barrier
    pool.front = 0;
    pool.size = 0;

    int falseM = 5000;

    startTime = omp_get_wtime();

    // Only one thread will need these vectors
    double startGpuMalloc, endGpuMalloc;
    startGpuMalloc = omp_get_wtime();

    // Common GPU-CPU
    Node *parents = (Node *)malloc(M * sizeof(Node));
    Node *children = (Node *)malloc(jobs * M * sizeof(Node));
    Node *stolenNodes = (Node *)malloc(5 * M * sizeof(Node));

    // GPU bounding functions data
    lb1_bound_data lbound1_d;
    int *p_times_d, *min_heads_d, *min_tails_d;
    lb1_alloc_gpu(&lbound1_d, lbound1, p_times_d, min_heads_d, min_tails_d, jobs, machines);

    lb2_bound_data lbound2_d;
    int *johnson_schedule_d, *lags_d, *machine_pairs_1_d, *machine_pairs_2_d, *machine_pair_order_d;
    lb2_alloc_gpu(&lbound2_d, lbound2, johnson_schedule_d, lags_d, machine_pairs_1_d, machine_pairs_2_d, machine_pair_order_d, jobs, machines);

    // Allocating parents vector on CPU and GPU
    // TODO: look single-GPU file remark!
    Node *parents_d;
    if (cpuID % NB_THREADS_GPU == 0)
      cudaMalloc((void **)&parents_d, M * sizeof(Node));

    int *sumOffSets = (int *)malloc(M * sizeof(int));
    int *sumOffSets_d;
    cudaMalloc((void **)&sumOffSets_d, M * sizeof(int));

    // Allocating bounds vector on CPU and GPU
    int *nodeIndex = (int *)malloc((jobs * M) * sizeof(int));
    int *nodeIndex_d;
    cudaMalloc((void **)&nodeIndex_d, (jobs * M) * sizeof(int));

    // Allocating bounds vector on CPU and GPU
    int *bounds = (int *)malloc((jobs * M) * sizeof(int));
    int *bounds_d;
    cudaMalloc((void **)&bounds_d, (jobs * M) * sizeof(int));

    endGpuMalloc = omp_get_wtime();
    timeGpuMalloc[cpuID] = endGpuMalloc - startGpuMalloc;

    int counter = 0;

    while (1)
    {
      // Each task gets its parenst nodes from the pool

      int poolSize; // one variable per each?

      if (cpuID % NB_THREADS_GPU != 0)
        poolSize = popBackBulk(pool_loc, m, falseM, parents, 1);
      else
        poolSize = popBackBulk(pool_loc, m, M, parents, 1);

      if (poolSize > 0)
      {
        if (cpuID % NB_THREADS_GPU != 0)
        {
          // CPU computation
          pushBackBulk(&parentsPool, parents, poolSize);
          int hasWork = 1;
          while (hasWork)
          {
            hasWork = 0;
            Node parent = popBackFree(&parentsPool, &hasWork);
            if (hasWork)
            {
              if (taskState == IDLE)
              {
                taskState = BUSY;
                atomic_store(&eachTaskState[cpuID], BUSY);
              }
              decompose(jobs, cpulb, best, lbound1, lbound2, parent, &tree, &sol, &childrenPool);
            }
          }
          eachExpTree[cpuID] = tree;
          eachExpSol[cpuID] = sol;

          if (childrenPool.size > 0)
          {
            int childrenSize = popBackBulkFree(&childrenPool, 1, childrenPool.size, children, 1);
            pushBackBulk(pool_loc, children, childrenSize);
          }
        }
        else
        {
          // GPU computation
          if (taskState == IDLE)
          {
            taskState = BUSY;
            atomic_store(&eachTaskState[cpuID], BUSY);
          }

          startGpuCpy = omp_get_wtime();
          int sum = 0;
          int diff;
          int i, j;
          for (i = 0; i < poolSize; i++)
          {
            diff = jobs - parents[i].depth;
            for (j = 0; j < diff; j++)
              nodeIndex[j + sum] = i;
            sum += diff;
            sumOffSets[i] = sum;
          }

          int numBounds = sum;
          int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);

          cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);
          cudaMemcpy(sumOffSets_d, sumOffSets, poolSize * sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(nodeIndex_d, nodeIndex, numBounds * sizeof(int), cudaMemcpyHostToDevice);
          endGpuCpy = omp_get_wtime();
          timeGpuCpy[cpuID] += endGpuCpy - startGpuCpy;
          // numBounds is the 'size' of the problem
          startGpuKer = omp_get_wtime();
          evaluate_gpu(jobs, lb, numBounds, nbBlocks, poolSize, best, lbound1_d, lbound2_d, parents_d, bounds_d, sumOffSets_d, nodeIndex_d);
          cudaDeviceSynchronize();
          endGpuKer = omp_get_wtime();
          timeGpuKer[cpuID] += endGpuKer - startGpuKer;

          startGpuCpy = omp_get_wtime();
          cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);
          endGpuCpy = omp_get_wtime();
          timeGpuCpy[cpuID] += endGpuCpy - startGpuCpy;

          // Each task generates and inserts its children nodes to the pool.
          startGenChild = omp_get_wtime();
          if (best_l != *best)
            checkBest(&best_l, best, &bestLock);
          int indexChildren;
          generate_children(parents, children, poolSize, jobs, bounds, &tree, &sol, &best_l, &indexChildren);
          if (best_l != *best)
            checkBest(&best_l, best, &bestLock);
          endGenChild = omp_get_wtime();
          timeGenChild[cpuID] += endGenChild - startGenChild;

          startPoolOps = omp_get_wtime();
          pushBackBulk(pool_loc, children, indexChildren);
          genChildGPU[cpuID] += indexChildren;
          endPoolOps = omp_get_wtime();
          timePoolOps[cpuID] += endPoolOps - startPoolOps;
          eachExpTree[cpuID] = tree;
          eachExpSol[cpuID] = sol;

          if (counter % 10000 == 0)
          {
            endTime = omp_get_wtime();
            double t2 = endTime - startTime;
            unsigned long long int partialExpTree = 0, partialExpSol = 0;
            for (int m = 0; m < NB_THREADS_MAX; m++)
            {
              partialExpTree += eachExpTree[m];
              partialExpSol += eachExpSol[m];
            }
            printf("Counter[%d] GPU[%d]: Tree[%llu] Sol[%llu]\n Pool: size[%d] capacity[%d] poolSize[%d]\n Timer: Total[%f] cudaMemcpy[%f] cudaMalloc[%f] kernelCall[%f] generateChildren[%f]\n",
                   counter, cpuID, partialExpTree, partialExpSol, pool_loc->size, pool_loc->capacity, poolSize, t2, timeGpuCpy[cpuID], timeGpuMalloc[cpuID], timeGpuKer[cpuID], timeGenChild[cpuID]);
          }
          counter++;
        }
      }
      else
      {
        if (ws == 0)
          break;
        else
        {
          // Local work stealing
          startGpuIdle = omp_get_wtime();
          int tries = 0;
          bool steal = false;
          int victims[NB_THREADS_MAX];
          permute(victims, NB_THREADS_MAX);
          bool expected;

          while (tries < NB_THREADS_MAX && steal == false)
          { // WS0 loop
            const int victimID = victims[tries];

            if (victimID != cpuID)
            { // if not me
              SinglePool_atom *victim;
              victim = &multiPool[victimID];
              nbSteals++;
              int nn = 0;
              while (nn < 10)
              { // WS1 loop
                expected = false;
                if (atomic_compare_exchange_strong(&(victim->lock), &expected, true))
                { // get the lock
                  int size = victim->size;
                  // printf("Victim[%d]->size[%d] & size[%d]\n", victimID, victim->size, size);
                  if (size >= 2 * m)
                  {
                    // printf("Victim[%d]->size[%d] & size[%d]\n", victimID, victim->size, size);
                    //  Higher value for parameter M allows real application of steal-half strategy
                    // int stolenNodesSize = popBackBulkFree(victim, m, 5 * M, stolenNodes, 2); // ratio is 2
                    // Node *p = popFrontBulkFree(victim, m, M, &nodeSize, perc);
                    int stolenNodesSize;
                    if (cpuID % NB_THREADS_GPU == 0)
                      stolenNodesSize = popBackBulkFree(victim, m, 5 * M, stolenNodes, 2); // ratio is 2
                    else
                      stolenNodesSize = popBackBulkFree(victim, m, 4 * falseM, stolenNodes, 2); // ratio is 2

                    if (stolenNodesSize == 0)
                    { // safety check
                      printf("\nThread [%d] DEADCODE Victim[%d]->size[%d] & size[%d]\n", cpuID, victimID, victim->size, size);
                      atomic_store(&(victim->lock), false); // reset lock
                      exit(-1);
                    }

                    startPoolOps = omp_get_wtime();
                    pushBackBulk(pool_loc, stolenNodes, stolenNodesSize); // necessary lock operation
                    endPoolOps = omp_get_wtime();
                    timePoolOps[cpuID] += endPoolOps - startPoolOps;

                    steal = true;
                    nbSSteals++;
                    atomic_store(&(victim->lock), false); // reset lock
                    goto WS0;                             // Break out of WS0 loop
                  }
                  atomic_store(&(victim->lock), false); // reset lock
                  break;                                // Break out of WS1 loop
                }
                nn++;
              }
            }
            tries++;
          }
        WS0:
          endGpuIdle = omp_get_wtime();
          timeGpuIdle[cpuID] += endGpuIdle - startGpuIdle;
          if (steal == false) // Termination Detection
          {
            startTermination = omp_get_wtime();
            nbTerminationGPU[cpuID]++;
            if (taskState == BUSY)
            {
              taskState = IDLE;
              atomic_store(&eachTaskState[cpuID], IDLE);
            }
            if (allIdle(eachTaskState, NB_THREADS_MAX, &allTasksIdleFlag))
            {
              endTermination = omp_get_wtime();
              timeTermination[cpuID] += endTermination - startTermination;
              break; // Break from GPU-accelerated bounding phase while
            }
            endTermination = omp_get_wtime();
            timeTermination[cpuID] += endTermination - startTermination;
            continue;
          }
          else
          {
            continue;
          }
        }
      }
    }

    free(parents);
    free(children);
    free(stolenNodes);

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
    free(bounds);
    free(sumOffSets);
    free(nodeIndex);

#pragma omp critical
    {
      // printf("Thread[%d] best_l[%d] best[%d]\n", cpuID, best_l, *best);
      nbStealsGPU[cpuID] = nbSteals;
      nbSStealsGPU[cpuID] = nbSSteals;
      expTreeGPU[cpuID] = tree;
      expSolGPU[cpuID] = sol;
      *exploredTree += expTreeGPU[cpuID];
      *exploredSol += expSolGPU[cpuID];
      //*best = MIN(*best, best_l);
      const int poolLocSize = pool_loc->size;
      for (int i = 0; i < poolLocSize; i++)
      {
        int hasWork = 0;
        pushBackFree(&pool, popBackFree(pool_loc, &hasWork));
        if (!hasWork)
          break;
      }
    }
    deleteSinglePool_atom(pool_loc);
  } // End of parallel region

  *best = best_l;
  endTime = omp_get_wtime();
  double t2 = endTime - startTime;

  printf("\nSearch on Parallel CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t2);

  /*
    Step 3: We complete the depth-first search on CPU.
  */

  startTime = omp_get_wtime();
  while (1)
  {
    int hasWork = 0;
    Node parent = popBackFree(&pool, &hasWork);
    if (!hasWork)
      break;
    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }

  // Freeing memory for structs common to all steps
  deleteSinglePool_atom(&pool);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);

  endTime = omp_get_wtime();
  double t3 = endTime - startTime;
  *elapsedTime = t1 + t2 + t3;
  printf("\nSearch on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t3);

  printf("\nExploration terminated.\n");
}

int main(int argc, char *argv[])
{
  srand(time(NULL));
  int version = 2; // Multi-GPU version is code 2
  // Parallel PFSP only uses: inst, lb, ub, m, M, D, ws
  int inst, lb, ub, m, M, D, ws, LB, commSize = 1; // commSize is an artificial variable here
  double perc;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &D, &ws, &LB, &perc);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  print_settings(inst, machines, jobs, ub, lb, D, ws, commSize, LB, version);

  int nb_procs = omp_get_num_procs();
  int MAX_GPU = 8;
  int NB_THREADS_GPU = (nb_procs / MAX_GPU);
  int NB_THREADS_MAX = D * NB_THREADS_GPU;

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0, exploredSol = 0;
  //   unsigned long long int expTreeGPU[D], expSolGPU[D], genChildGPU[D], nbStealsGPU[D], nbSStealsGPU[D], nbTerminationGPU[D];
  unsigned long long int expTreeCPU[NB_THREADS_MAX], expSolCPU[NB_THREADS_MAX], genChildCPU[NB_THREADS_MAX], nbStealsCPU[NB_THREADS_MAX], nbSStealsCPU[NB_THREADS_MAX], nbTerminationCPU[NB_THREADS_MAX];

  double elapsedTime = 0;
  // double timeGpuCpy[D], timeGpuMalloc[D], timeGpuKer[D], timeGenChild[D], timePoolOps[D], timeGpuIdle[D], timeTermination[D];
  double timeGpuCpy[NB_THREADS_MAX], timeGpuMalloc[NB_THREADS_MAX], timeCpuKer[NB_THREADS_MAX], timeGenChild[NB_THREADS_MAX], timePoolOps[NB_THREADS_MAX], timeCpuIdle[NB_THREADS_MAX], timeTermination[NB_THREADS_MAX];

  for (int i = 0; i < NB_THREADS_MAX; i++)
  {
    timeGpuCpy[i] = 0;
    timeGpuMalloc[i] = 0;
    timeCpuKer[i] = 0;
    timeGenChild[i] = 0;
    timePoolOps[i] = 0;
    timeCpuIdle[i] = 0;
    timeTermination[i] = 0;
    expTreeCPU[i] = 0;
    expSolCPU[i] = 0;
    genChildCPU[i] = 0;
    nbStealsCPU[i] = 0;
    nbSStealsCPU[i] = 0;
    nbTerminationCPU[i] = 0;
  }

  pfsp_search(inst, lb, m, M, D, perc, ws, &optimum, &exploredTree, &exploredSol, &elapsedTime,
              expTreeCPU, expSolCPU, genChildCPU, nbStealsCPU, nbSStealsCPU, nbTerminationCPU,
              timeGpuCpy, timeGpuMalloc, timeCpuKer, timeGenChild, timePoolOps, timeCpuIdle, timeTermination);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  print_results_file_multi_gpu(inst, lb, NB_THREADS_MAX, ws, optimum, m, M, exploredTree, exploredSol, elapsedTime,
                               expTreeCPU, expSolCPU, genChildCPU, nbStealsCPU, nbSStealsCPU, nbTerminationCPU,
                               timeGpuCpy, timeGpuMalloc, timeCpuKer, timeGenChild, timePoolOps, timeCpuIdle, timeTermination);

  return 0;
}
