/*
  Multi-GPU B&B to solve Taillard instances of the PFSP in C+OpenMP+CUDA.
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
#include <omp.h>
#include <cuda_runtime.h>
#include <stdatomic.h>

#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"
#include "lib/PFSP_gpu_lib.cuh"
#include "lib/PFSP_lib.h"
#include "lib/Pool_atom.h"
#include "../common/util.h"

/******************************************************************************
Statistics functions
*****************************************************************************/
void print_results_file(const int inst, const int machines, const int jobs, const int lb, const int D, int ws, const int optimum,
                        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
                        unsigned long long int *expTreeGPU, unsigned long long int *expSolGPU, unsigned long long int *genChildren,
                        unsigned long long int *nStealsGPU, unsigned long long int *nSStealsGPU, unsigned long long int *nTerminationGPU,
                        double *timeCudaMemCpy, double *timeCudaMalloc, double *timeKernelCall, double *timeIdle, double *timeTermination,
                        double *timePoolOps, double *timeGenChildren)
{
  double maxTimeIdle, maxTimeTermination, maxTimeCudaMalloc, maxCudaMemcpy;
  maxTimeIdle = get_max(timeIdle, D);
  maxTimeTermination = get_max(timeTermination, D);
  maxTimeCudaMalloc = get_max(timeCudaMalloc, D);
  maxCudaMemcpy = get_max(timeCudaMemCpy, D);
  FILE *file;
  file = fopen("multigpu.dat", "a");
  fprintf(file, "\nMGPU-opt[%d]WS[%d] ta%d lb%d Time[%.4f] Max(IdleTime[%.4f]/Termination[%.4f]/Malloc[%.4f]/MemCpy[%.4f]) Tree[%llu] Sol[%llu] Best[%d]\n", D, ws, inst, lb, timer, maxTimeIdle, maxTimeTermination, maxTimeCudaMalloc, maxCudaMemcpy, exploredTree, exploredSol, optimum);
  fprintf(file, "Explored Nodes per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%llu ", expTreeGPU[i]);
    else
      fprintf(file, "%llu\n", expTreeGPU[i]);
  }
  if (lb != 2)
  {
    fprintf(file, "Explored Solutions per GPU: ");
    for (int i = 0; i < D; i++)
    {
      if (i != D - 1)
        fprintf(file, "%llu ", expSolGPU[i]);
      else
        fprintf(file, "%llu\n", expSolGPU[i]);
    }
  }
  fprintf(file, "Work Stealing (Attempt/Succesful) / Termination Checks per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%llu/%llu/%llu ", nStealsGPU[i], nSStealsGPU[i], nTerminationGPU[i]);
    else
      fprintf(file, "%llu/%llu/%llu\n", nStealsGPU[i], nSStealsGPU[i], nTerminationGPU[i]);
  }
  fprintf(file, "Time kernelCall per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%.4f ", timeKernelCall[i]);
    else
      fprintf(file, "%.4f\n", timeKernelCall[i]);
  }
  fclose(file);

  file = fopen("multigpu_detail.dat", "a");
  fprintf(file, "\nMGPU-opt[%d]WS[%d] ta%d lb%d Time[%.4f] Tree[%llu] Sol[%llu] Best[%d]\n", D, ws, inst, lb, timer, exploredTree, exploredSol, optimum);
  fprintf(file, "Explored Nodes per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%llu ", expTreeGPU[i]);
    else
      fprintf(file, "%llu\n", expTreeGPU[i]);
  }
  fprintf(file, "Explored Solutions per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%llu ", expSolGPU[i]);
    else
      fprintf(file, "%llu\n", expSolGPU[i]);
  }
  fprintf(file, "Attempt Work Stealing per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%llu ", nStealsGPU[i]);
    else
      fprintf(file, "%llu\n", nStealsGPU[i]);
  }
  fprintf(file, "Succesful Work Stealing per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%llu ", nSStealsGPU[i]);
    else
      fprintf(file, "%llu\n", nSStealsGPU[i]);
  }
  fprintf(file, "Termination Checks per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%llu ", nTerminationGPU[i]);
    else
      fprintf(file, "%llu\n", nTerminationGPU[i]);
  }
  fprintf(file, "Time cudaMemcpy per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%.4f ", timeCudaMemCpy[i]);
    else
      fprintf(file, "%.4f\n", timeCudaMemCpy[i]);
  }
  fprintf(file, "Time cudaMalloc per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%.4f ", timeCudaMalloc[i]);
    else
      fprintf(file, "%.4f\n", timeCudaMalloc[i]);
  }
  fprintf(file, "Time kernelCall per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%.4f ", timeKernelCall[i]);
    else
      fprintf(file, "%.4f\n", timeKernelCall[i]);
  }
  fprintf(file, "Time Idle per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%.4f ", timeIdle[i]);
    else
      fprintf(file, "%.4f\n", timeIdle[i]);
  }
  fprintf(file, "Time Termination per GPU: ");
  for (int i = 0; i < D; i++)
  {
    if (i != D - 1)
      fprintf(file, "%.4f ", timeTermination[i]);
    else
      fprintf(file, "%.4f\n\n", timeTermination[i]);
  }
  fclose(file);
  return;
}

// Multi-GPU PFSP search
void pfsp_search(const int inst, const int lb, const int m, const int M, const int D, const double perc, int ws, int *best,
                 unsigned long long int *exploredTree, unsigned long long int *exploredSol, double *elapsedTime, unsigned long long int *expTreeGPU,
                 unsigned long long int *expSolGPU, unsigned long long int *genChildren, unsigned long long int *nStealsGPU, unsigned long long int *nSStealsGPU,
                 unsigned long long int *nTerminationGPU, double *timeCudaMemCpy, double *timeCudaMalloc, double *timeKernelCall, double *timeIdle,
                 double *timeTermination, double *timePoolOps, double *timeGenChildren)
{
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
  _Atomic bool allTasksIdleFlag = false;
  _Atomic bool eachTaskState[D]; // one task per GPU
  for (int i = 0; i < D; i++)
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

  while (pool.size < D * m)
  {
    // CPU side
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

  unsigned long long int eachExploredTree[D], eachExploredSol[D];
  int eachBest[D];

  const int poolSize = pool.size;
  const int c = poolSize / D;
  const int l = poolSize - (D - 1) * c;
  const int f = pool.front;

  pool.front = 0;
  pool.size = 0;
  SinglePool_atom multiPool[D];
  for (int i = 0; i < D; i++)
    initSinglePool_atom(&multiPool[i]);

  startTime = omp_get_wtime();

  double timeDevice[D];

// TODO: implement reduction using omp directives
#pragma omp parallel num_threads(D) shared(eachExploredTree, eachExploredSol, eachBest, eachTaskState, allTasksIdleFlag, pool, multiPool,                 \
                                               jobs, machines, lbound1, lbound2, lb, m, M, D, perc, ws, best, exploredTree, exploredSol,                  \
                                               elapsedTime, expTreeGPU, expSolGPU, genChildren, nStealsGPU, nSStealsGPU, nTerminationGPU, timeCudaMemCpy, \
                                               timeCudaMalloc, timeKernelCall, timeIdle, timeTermination, timeDevice)
  // for (int gpuID = 0; gpuID < D; gpuID++)
  {
    double startCudaMemCpy, endCudaMemCpy, startCudaMalloc, endCudaMalloc, startKernelCall, endKernelCall, startTimePoolOps, endTimePoolOps,
        startTimeIdle, endTimeIdle, startTermination, endTermination, startGenChildren, endGenChildren, startSetDevice, endSetDevice;
    int nSteal = 0, nSSteal = 0;
    int gpuID = omp_get_thread_num();
    startSetDevice = omp_get_wtime();
    cudaSetDevice(gpuID);
    endSetDevice = omp_get_wtime();
    double timeSetDevice = endSetDevice - startSetDevice;
    timeDevice[gpuID] = timeSetDevice;
    // printf("GPU[%d] Time to set device: %f\n", gpuID, timeSetDevice);

    // startPool = omp_get_wtime();
    unsigned long long int tree = 0, sol = 0;
    SinglePool_atom *pool_loc;
    pool_loc = &multiPool[gpuID];
    int best_l = *best;
    bool taskState = BUSY;

    // each task gets its chunk
    for (int i = 0; i < c; i++)
    {
      pool_loc->elements[i] = pool.elements[gpuID + f + i * D];
    }
    pool_loc->size += c;
    if (gpuID == D - 1)
    {
      for (int i = c; i < l; i++)
      {
        pool_loc->elements[i] = pool.elements[(D * c) + f + i - c];
      }
      pool_loc->size += l - c;
    }
    // endPool = omp_get_wtime();
    // double timePool = endPool - startPool;
    // printf("GPU[%d] Time to redistribute pool: %f\n", gpuID, timePool);

    startCudaMalloc = omp_get_wtime();
    // GPU bounding functions data
    lb1_bound_data lbound1_d;
    int *p_times_d, *min_heads_d, *min_tails_d;
    lb1_alloc_gpu(&lbound1_d, lbound1, p_times_d, min_heads_d, min_tails_d, jobs, machines);

    lb2_bound_data lbound2_d;
    int *johnson_schedule_d, *lags_d, *machine_pairs_1_d, *machine_pairs_2_d, *machine_pair_order_d;
    lb2_alloc_gpu(&lbound2_d, lbound2, johnson_schedule_d, lags_d, machine_pairs_1_d, machine_pairs_2_d, machine_pair_order_d, jobs, machines);

    // Allocating parents vector on CPU and GPU
    Node *parents = (Node *)malloc(M * sizeof(Node));
    Node *children = (Node *)malloc(jobs * M * sizeof(Node));
    Node *parents_d;
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

    endCudaMalloc = omp_get_wtime();
    timeCudaMalloc[gpuID] = endCudaMalloc - startCudaMalloc;

    int indexChildren;

    while (1)
    {
      /*
        Each task gets its parenst nodes from the pool
      */
      startTimePoolOps = omp_get_wtime();
      int poolSize = popBackBulk(pool_loc, m, M, parents);
      endTimePoolOps = omp_get_wtime();
      timePoolOps[gpuID] += endTimePoolOps - startTimePoolOps;

      if (poolSize > 0)
      {
        if (taskState == IDLE)
        {
          taskState = BUSY;
          atomic_store(&eachTaskState[gpuID], BUSY);
        }

        startCudaMemCpy = omp_get_wtime();
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

        const int numBounds = sum;
        const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);

        cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);
        cudaMemcpy(sumOffSets_d, sumOffSets, poolSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(nodeIndex_d, nodeIndex, numBounds * sizeof(int), cudaMemcpyHostToDevice);
        endCudaMemCpy = omp_get_wtime();
        timeCudaMemCpy[gpuID] += endCudaMemCpy - startCudaMemCpy;
        // numBounds is the 'size' of the problem
        startKernelCall = omp_get_wtime();
        evaluate_gpu(jobs, lb, numBounds, nbBlocks, poolSize, best, lbound1_d, lbound2_d, parents_d, bounds_d, sumOffSets_d, nodeIndex_d);
        // evaluate_gpu(jobs, lb, numBounds, nbBlocks, &best_l, lbound1_d, lbound2_d, parents_d, bounds_d);
        cudaDeviceSynchronize();
        endKernelCall = omp_get_wtime();
        timeKernelCall[gpuID] += endKernelCall - startKernelCall;

        startCudaMemCpy = omp_get_wtime();
        cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);
        endCudaMemCpy = omp_get_wtime();
        timeCudaMemCpy[gpuID] += endCudaMemCpy - startCudaMemCpy;

        /*
          Each task generates and inserts its children nodes to the pool.
        */
        startGenChildren = omp_get_wtime();
        generate_children(parents, children, poolSize, jobs, bounds, &tree, &sol, &best_l, pool_loc, &indexChildren);
        endGenChildren = omp_get_wtime();
        timeGenChildren[gpuID] += endGenChildren - startGenChildren;

        startTimePoolOps = omp_get_wtime();
        pushBackBulk(pool_loc, children, indexChildren);
        endTimePoolOps = omp_get_wtime();
        timePoolOps[gpuID] += endTimePoolOps - startTimePoolOps;

        genChildren[gpuID] += indexChildren;
      }
      else
      {
        if (ws == 0)
          break;
        else
        {
          // Local work stealing
          startTimeIdle = omp_get_wtime();
          int tries = 0;
          bool steal = false;
          int victims[D];
          permute(victims, D);
          bool expected;

          while (tries < D && steal == false)
          { // WS0 loop
            const int victimID = victims[tries];

            if (victimID != gpuID)
            { // if not me
              SinglePool_atom *victim;
              victim = &multiPool[victimID];
              nSteal++;
              int nn = 0;
              while (nn < 10)
              { // WS1 loop
                expected = false;
                if (atomic_compare_exchange_strong(&(victim->lock), &expected, true))
                { // get the lock
                  int size = victim->size;
                  int nodeSize = 0;

                  if (size >= 2 * m)
                  {
                    Node *p = popBackBulkFree(victim, m, M, &nodeSize);
                    // Node *p = popFrontBulkFree(victim, m, M, &nodeSize, perc);

                    if (nodeSize == 0)
                    {                                       // safety check
                      atomic_store(&(victim->lock), false); // reset lock
                      printf("\nDEADCODE\n");
                      exit(-1);
                    }

                    startTimePoolOps = omp_get_wtime();
                    pushBackBulk(pool_loc, p, nodeSize); // atomic_store inside
                    endTimePoolOps = omp_get_wtime();
                    timePoolOps[gpuID] += endTimePoolOps - startTimePoolOps;

                    steal = true;
                    nSSteal++;
                    atomic_store(&(victim->lock), false); // reset lock
                    // endTimeIdle = omp_get_wtime();
                    // timeIdle[gpuID] += endTimeIdle - startTimeIdle;
                    goto WS0; // Break out of WS0 loop
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
          endTimeIdle = omp_get_wtime();
          timeIdle[gpuID] += endTimeIdle - startTimeIdle;
          if (steal == false)
          {
            startTermination = omp_get_wtime();
            // termination
            nTerminationGPU[gpuID]++;
            if (taskState == BUSY)
            {
              taskState = IDLE;
              atomic_store(&eachTaskState[gpuID], IDLE);
            }
            if (allIdle(eachTaskState, D, &allTasksIdleFlag))
            {
              endTermination = omp_get_wtime();
              timeTermination[gpuID] += endTermination - startTermination;
              break;
            }
            endTermination = omp_get_wtime();
            timeTermination[gpuID] += endTermination - startTermination;
            continue;
          }
          else
          {
            continue;
          }
        }
      }
    }

    // Freeing variables from OpenMP environment
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
    free(parents);
    free(bounds);

#pragma omp critical
    {
      const int poolLocSize = pool_loc->size;
      for (int i = 0; i < poolLocSize; i++)
      {
        int hasWork = 0;
        pushBackFree(&pool, popBackFree(pool_loc, &hasWork));
        if (!hasWork)
          break;
      }
    }

    eachExploredTree[gpuID] = tree;
    eachExploredSol[gpuID] = sol;
    eachBest[gpuID] = best_l;

    expTreeGPU[gpuID] = tree;
    expSolGPU[gpuID] = sol;
    nStealsGPU[gpuID] = nSteal;
    nSStealsGPU[gpuID] = nSSteal;

    deleteSinglePool_atom(pool_loc);
  } // End of parallel region

  endTime = omp_get_wtime();
  double t2 = endTime - startTime;

  double maxDevice = get_max(timeDevice, D);
  t2 -= maxDevice;

  for (int i = 0; i < D; i++)
  {
    *exploredTree += eachExploredTree[i];
    *exploredSol += eachExploredSol[i];
  }
  *best = findMin(eachBest, D);

  printf("\nSearch on GPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t2);
  printf("Workload per GPU: ");
  for (int gpuID = 0; gpuID < D; gpuID++)
    printf("%.2f ", (double)100 * eachExploredTree[gpuID] / ((double)*exploredTree));
  printf("\n");
  printf("Time in generate_children : ");
  for (int i = 0; i < D; i++)
    printf("%.2f ", timeGenChildren[i]);
  printf("\n");

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
  // Multi-GPU PFSP only uses: inst, lb, ub, m, M, D, ws
  int inst, lb, ub, m, M, D, ws, LB, commSize = 1; // commSize is an artificial variable here
  double perc;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &D, &ws, &LB, &perc);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  print_settings(inst, machines, jobs, ub, lb, D, ws, commSize, LB, version);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;
  unsigned long long int expTreeGPU[D], expSolGPU[D], genChildren[D], nStealsGPU[D], nSStealsGPU[D], nTerminationGPU[D];

  double elapsedTime;
  double timeCudaMemCpy[D], timeCudaMalloc[D], timeKernelCall[D], timeIdle[D], timeTermination[D], timePoolOps[D], timeGenChildren[D];

  for (int i = 0; i < D; i++)
  {
    timeCudaMemCpy[i] = 0;
    timeCudaMalloc[i] = 0;
    timeKernelCall[i] = 0;
    timeIdle[i] = 0;
    timeTermination[i] = 0;
    timePoolOps[i] = 0;
    timeGenChildren[i] = 0;
    nTerminationGPU[i] = 0;
    genChildren[i] = 0;
  }

  pfsp_search(inst, lb, m, M, D, perc, ws, &optimum, &exploredTree, &exploredSol, &elapsedTime,
              expTreeGPU, expSolGPU, genChildren, nStealsGPU, nSStealsGPU, nTerminationGPU, timeCudaMemCpy, timeCudaMalloc,
              timeKernelCall, timeIdle, timeTermination, timePoolOps, timeGenChildren);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  print_results_file(inst, machines, jobs, lb, D, ws, optimum, exploredTree, exploredSol, elapsedTime,
                     expTreeGPU, expSolGPU, genChildren, nStealsGPU, nSStealsGPU, nTerminationGPU, timeCudaMemCpy, timeCudaMalloc,
                     timeKernelCall, timeIdle, timeTermination, timePoolOps, timeGenChildren);

  return 0;
}
