/*
Multi-GPU B&B to solve Taillard instances of the PFSP based on MPI+CUDA written in C language
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
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

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

/******************************************************************************
PFSP MPI Library
******************************************************************************/

void create_mpi_node_type(MPI_Datatype *mpi_node_type)
{
  int blocklengths[3] = {1, 1, MAX_JOBS};
  MPI_Aint offsets[3];
  offsets[0] = offsetof(Node, depth);
  offsets[1] = offsetof(Node, limit1);
  offsets[2] = offsetof(Node, prmu);

  MPI_Datatype types[3] = {MPI_INT16_T, MPI_INT16_T, MPI_INT16_T};
  MPI_Type_create_struct(3, blocklengths, offsets, types, mpi_node_type);
  MPI_Type_commit(mpi_node_type);
}

void check_win_error(int err, int *vector)
{
  if (err != MPI_SUCCESS || vector == NULL)
  {
    char err_string[MPI_MAX_ERROR_STRING];
    int err_length;
    MPI_Error_string(err, err_string, &err_length);
    fprintf(stderr, "MPI_Win_allocate failed: %s\n", err_string);
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  return;
}

/*******************************************************************************
Implementation of the parallel Multi-GPU C+MPI+CUDA PFSP search.
*******************************************************************************/
void pfsp_search(const int inst, const int lb, const int m, const int M, const int T, const int D, const int C, const int L, int MPIRank, int commSize,
                 double perc, int *best, unsigned long long int *exploredTree, unsigned long long int *exploredSol, double *elapsedTime,
                 unsigned long long int *all_expTreeGPU, unsigned long long int *all_expSolGPU, unsigned long long int *all_genChildGPU,
                 unsigned long long int *all_nbStealsGPU, unsigned long long int *all_nbSStealsGPU, unsigned long long int *all_nbTerminationGPU,
                 unsigned long long int *nbSDistLoadBal, double *all_timeGpuCpy, double *all_timeGpuMalloc, double *all_timeGpuKer, double *all_timeGenChild,
                 double *all_timePoolOps, double *all_timeGpuIdle, double *all_timeTermination, double *timeLoadBal)
{
  // New MPI data type corresponding to Node
  MPI_Datatype myNode;
  create_mpi_node_type(&myNode);
  int NB_THREADS_MAX = commSize; // For now, the amount of MPI processes

  // Initialize problem parameters
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  // Starting pool
  Node root;
  initRoot(&root, jobs);
  SinglePool_atom pool;
  initSinglePool_atom(&pool);
  pushBack(&pool, root);

  // Global timer
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
    Step 1: We perform a partial BFS on every MPI process to create
    a sufficiently large amount of work for GPU computation.
  */
  while (pool.size < commSize * m)
  {
    int hasWork = 0;
    Node parent = popFrontFree(&pool, &hasWork);
    if (!hasWork)
      break;
    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }

  endTime = omp_get_wtime();
  double t1, t1Temp = endTime - startTime;
  MPI_Reduce(&t1Temp, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (MPIRank == 0)
  {
    printf("\nInitial search on CPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("Elapsed time: %f [s]\n", t1);
  }
  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */

  SinglePool_atom pool_lloc;
  initSinglePool_atom(&pool_lloc);

  // each MPI process gets its chunk
  roundRobin_distribution(&pool_lloc, &pool, MPIRank, commSize);
  pool.front = 0;
  pool.size = 0;

  // For every MPI process
  unsigned long long int eachLocaleExploredTree = 0, eachLocaleExploredSol = 0;
  int eachLocaleBest = *best;

  // One per compute node (needs to be communicated between processes)
  _Atomic bool localeState = false;
  // WARNING: For termination detection of distributed step
  _Atomic bool allLocalesIdleFlag = false;

  // One-Sided Shared-Memory Windows

  // TODO: Window Info
  // MPI_Info info_nodes, info_requests;
  int *steal_request;
  MPI_Win win_requests;
  int err = MPI_Win_allocate(commSize * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &steal_request, &win_requests);
  check_win_error(err, steal_request);

  int *steal_sizes;
  MPI_Win win_sizes;
  err = MPI_Win_allocate(commSize * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &steal_sizes, &win_sizes);
  check_win_error(err, steal_sizes);

  // int capacity_per_proc = 300000;
  // Node *steal_nodes;
  // MPI_Win win_nodes;
  // err = MPI_Win_allocate(commSize * capacity_per_proc * sizeof(Node), sizeof(Node), MPI_INFO_NULL, MPI_COMM_WORLD, &steal_nodes, &win_nodes);
  // if (err != MPI_SUCCESS || steal_nodes == NULL)
  // {
  //   char err_string[MPI_MAX_ERROR_STRING];
  //   int err_length;
  //   MPI_Error_string(err, err_string, &err_length);
  //   fprintf(stderr, "MPI_Win_allocate for nodes failed: %s\n", err_string);
  //   MPI_Abort(MPI_COMM_WORLD, err);
  // }

  for (int i = 0; i < commSize; i++)
  {
    // From 0 to 'commSize - 1' indicates which MPIRank contributed with work
    // '-2' indicates the need of work
    steal_request[i] = -1; // BUSY
    steal_sizes[i] = 0;
  }

  // MPI_Win_sync(win_sizes);
  // MPI_Win_sync(win_requests);
  // MPI_Win_sync(win_nodes);
  // MPI_Barrier(MPI_COMM_WORLD);

  // EVENTUAL USEFUL VARIABLES
  cudaSetDevice(MPIRank);

  int nSteal = 0, nSSteal = 0;
  int stolen = 0;

  // int amount_nodes = 0;
  // double time1, time2, time1Idle, time2Idle, put_sizes = 0, put_request = 0, put_nodes = 0, compare_request = 0, sync_sizes = 0,
  //   time1GPUWork, time2GPUWork, GPUWork = 0, GPUTime = 0, IdleTime = 0, cpyParents = 0, cpyBounds = 0, cpyKernel = 0;
  // int best_l = *best;
  // bool expected = false;

  // GPU bounding functions data
  lb1_bound_data lbound1_d;
  int *p_times_d, *min_heads_d, *min_tails_d;
  lb1_alloc_gpu(&lbound1_d, lbound1, p_times_d, min_heads_d, min_tails_d, jobs, machines);

  lb2_bound_data lbound2_d;
  int *johnson_schedule_d, *lags_d, *machine_pairs_1_d, *machine_pairs_2_d, *machine_pair_order_d;
  lb2_alloc_gpu(&lbound2_d, lbound2, johnson_schedule_d, lags_d, machine_pairs_1_d, machine_pairs_2_d, machine_pair_order_d, jobs, machines);

  // Allocating parents vector on CPU and GPU
  Node *parents = (Node *)malloc(M * sizeof(Node));
  Node *stolenNodes = (Node *)malloc(5 * M * sizeof(Node));
  Node *children = (Node *)malloc(jobs * M * sizeof(Node));
  Node *parents_d;
  cudaMalloc((void **)&parents_d, M * sizeof(Node));

  // Allocating vectors GPU Thread Indexing
  int *sumOffSets = (int *)malloc(M * sizeof(int));
  int *sumOffSets_d;
  cudaMalloc((void **)&sumOffSets_d, M * sizeof(int));

  int *nodeIndex = (int *)malloc((jobs * M) * sizeof(int));
  int *nodeIndex_d;
  cudaMalloc((void **)&nodeIndex_d, (jobs * M) * sizeof(int));

  // Allocating bounds vector on CPU and GPU
  int *bounds = (int *)malloc((jobs * M) * sizeof(int));
  int *bounds_d;
  cudaMalloc((void **)&bounds_d, (jobs * M) * sizeof(int));

  printf("MPIRank[%d] is before GPU-accelerated while\n", MPIRank);
  while (1)
  {
    int poolSize = popBackBulk(&pool_lloc, m, M, parents, 1);
    // MPI_Win_sync(win_sizes);
    // MPI_Win_sync(win_requests);
    // MPI_Win_sync(win_nodes);
    // MPI_Win_flush_all(win_requests);
    if (poolSize > 0)
    {
      // time1GPUWork = omp_get_wtime();

      if (atomic_load(&localeState) == IDLE)
        atomic_store(&localeState, BUSY);

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

      evaluate_gpu(jobs, lb, numBounds, nbBlocks, poolSize, &eachLocaleBest, lbound1_d, lbound2_d, parents_d, bounds_d, sumOffSets_d, nodeIndex_d);
      cudaDeviceSynchronize();

      cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);

      // Each task generates and inserts its children nodes to the pool.
      int indexChildren;
      generate_children(parents, children, poolSize, jobs, bounds, &eachLocaleExploredTree, &eachLocaleExploredSol, &eachLocaleBest, &indexChildren);
      pushBackBulk(&pool_lloc, children, indexChildren);

      // Answer WS requests after a complete round of bounding+pruning+branching
      // MPI_Win_sync(win_requests);
      int tries = 0;
      int requests[commSize];
      permute(requests, commSize); // Introduce some randomness

      // DEBUG
      // for (int i = 0; i < commSize; i++)
      //   printf("Proc[%d] steal_request[%d] = %d\n", MPIRank, i, steal_request[i]);

      // TODO: Work Stealing with One-Sided Shared-Memory
      // while (tries < commSize && pool_lloc.size > 2 * m)
      // { // WS0 loop
      //   // MPI_Win_sync(win_sizes);
      //   // MPI_Win_sync(win_requests);
      //   // MPI_Win_sync(win_nodes);
      //   int requestID = requests[tries];

      //   if (steal_request[requestID] == -1 || requestID == MPIRank || steal_request[requestID] >= 0) // BUSY worker or MYself or Request attended
      //   {
      //     nSteal++;
      //     tries++;
      //     continue;
      //   }
      //   else if (steal_request[requestID] == -2) // Can send work
      //   {
      //     int expected = -2;       // Expected old value
      //     int new_value = MPIRank; // New value to set
      //     int old_value;           // Buffer for the old value

      //     // time1 = omp_get_wtime();
      //     //  Lock the window for atomic access
      //     MPI_Win_lock(MPI_LOCK_EXCLUSIVE, requestID, 0, win_requests);
      //     MPI_Compare_and_swap(&new_value, &expected, &old_value, MPI_INT, requestID, requestID, win_requests);
      //     // MPI_Win_sync(win_requests);
      //     MPI_Win_unlock(requestID, win_requests); // Unlock after atomic operation
      //     // nSSteal++;
      //     // time2 = omp_get_wtime();
      //     printf("Proc[%d] I did my MPI_compare_and_swap\n", MPIRank);
      //     // compare_request += time2 - time1;

      //     // If successful, notify
      //     if (old_value == -2)
      //     {
      //       steal_request[requestID] = -1;
      //       // int amount_nodes;
      //       (pool_lloc.size > 2 * capacity_per_proc) ? (amount_nodes = capacity_per_proc) : (amount_nodes = pool_lloc.size / 2);
      //       stolen += amount_nodes;
      //       printf("Proc[%d] Is successful amount_nodes = %d\n", MPIRank, amount_nodes);
      //       // I send the nodes
      //       Node *nodes = popBackBulkFreeN(&pool_lloc, m, M, &amount_nodes);
      //       if (amount_nodes == 0)
      //       {
      //         printf("ERROR Pool_lloc size is 0 on PopBackBulkFreeN\n");
      //         exit(-1);
      //       }

      //       // time1 = omp_get_wtime();

      //       // MPI_Win_lock(MPI_LOCK_EXCLUSIVE, requestID, 0, win_nodes);
      //       // MPI_Put(nodes, amount_nodes * sizeof(Node), MPI_BYTE, requestID, MPIRank * capacity_per_proc, amount_nodes * sizeof(Node), MPI_BYTE, win_nodes);
      //       // // MPI_Win_flush(requestID, win_nodes);
      //       // MPI_Win_flush_all(win_nodes);
      //       // MPI_Win_sync(win_nodes);
      //       // MPI_Win_unlock(requestID, win_nodes); // Unlock after atomic operation

      //       // MPI_Win_sync(win_nodes);
      //       // time2 = omp_get_wtime();

      //       // put_nodes += time2 - time1;

      //       // Update size after sending nodes
      //       // Nodes arrive before size
      //       // Target process does not try to access nodes that still did not exist for him
      //       // time1 = omp_get_wtime();

      //       // THE ERROR IS HEREEEEEEEEEEE =================================/////////////////////////////////////////////////
      //       // AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHAAAAAAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHH/////
      //       // MPI_Win_lock(MPI_LOCK_EXCLUSIVE, requestID, 0, win_sizes);
      //       // MPI_Put(&amount_nodes, 1, MPI_INT, requestID, MPIRank, 1, MPI_INT, win_sizes);
      //       // MPI_Win_flush(requestID, win_sizes);
      //       // MPI_Win_flush_all(win_sizes);
      //       // MPI_Win_sync(win_sizes);
      //       // MPI_Win_unlock(requestID, win_sizes);

      //       // USING MPI_SEND instead
      //       MPI_Send(&amount_nodes, 1, MPI_INT, requestID, 2, MPI_COMM_WORLD);

      //       MPI_Send(nodes, amount_nodes * sizeof(Node), MPI_BYTE, requestID, 1, MPI_COMM_WORLD);

      //       // MPI_Win_sync(win_sizes);
      //       // time2 = omp_get_wtime();

      //       // put_sizes += time2 - time1;

      //       // pool_lloc.size -= amount_nodes;

      //       printf("After Put: Proc[%d] gave [%d] work to RequestProc[%d]\n", MPIRank, amount_nodes, requestID);
      //       nSSteal++;
      //       break; // Break While
      //     }
      //     tries++;
      //   }
      // }
      // // time2GPUWork = omp_get_wtime();
      // // GPUWork += time2GPUWork - time1GPUWork;
    }
    else
    {
      // ( (1 MPI process) ||  (L == 0) ) == no Work-Stealing
      if (commSize == 1 || L == 0)
        break;

      // bool remoteSteal = true;
      // // int tries = 0;

      // // '-2' indicates steal request
      // steal_request[MPIRank] = -2;
      // printf("Proc [%d] Before sending steal request\n", MPIRank);
      // // time1 = omp_get_wtime();
      // for (int j = 0; j < commSize; j++)
      // {
      //   if (j == MPIRank)
      //     continue; // Skip self
      //   MPI_Win_lock(MPI_LOCK_EXCLUSIVE, j, 0, win_requests);
      //   MPI_Put(&steal_request[MPIRank], 1, MPI_INT, j, MPIRank, 1, MPI_INT, win_requests);
      //   // MPI_Win_flush(j, win_requests);
      //   MPI_Win_flush_all(win_requests);
      //   MPI_Win_sync(win_requests);
      //   MPI_Win_unlock(j, win_requests);
      // }
      // // time2 = omp_get_wtime();
      // // put_request += time2 - time1;

      // // MPI_Win_sync(win_sizes);
      // // MPI_Win_sync(win_requests);
      // // MPI_Win_sync(win_nodes);

      // while (steal_request[MPIRank] == -2 || steal_request[MPIRank] >= 0)
      // {
      //   int sum = 0;
      //   for (int j = 0; j < commSize; j++) // We check if all of them are asking for work
      //     sum += steal_request[j];

      //   int responder = steal_request[MPIRank];

      //   // printf("Responder == %d, sum == %d\n", responder, sum);
      //   MPI_Win_sync(win_nodes);
      //   MPI_Win_sync(win_sizes);

      //   if (responder >= 0)
      //   {
      //     printf("Proc[%d] has answer from StolenProc[%d]\n", MPIRank, responder);
      //     printf("Proc[%d] got [%d] work from StolenProc[%d]\n", MPIRank, steal_sizes[responder], responder);
      //     MPI_Recv(&steal_sizes[responder], 1, MPI_INT, responder, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      //     for (int i = 0; i < commSize; i++)
      //       printf("Proc[%d] steal_request[%d] = %d after responder\n", MPIRank, i, steal_request[i]);

      //     MPI_Recv((steal_nodes + (responder * capacity_per_proc)), steal_sizes[responder] * sizeof(Node), MPI_BYTE, responder, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      //     int counter = 0;
      //     // time1 = omp_get_wtime();

      //     // while (steal_sizes[responder] == 0)
      //     // {
      //     //   //MPI_Win_sync(win_requests);
      //     //   //MPI_Win_sync(win_nodes);

      //     //   counter++;
      //     //   if (counter % 500 == 0)
      //     //     MPI_Win_sync(win_sizes);

      //     //   if (counter % 1000000 == 0)
      //     //   {
      //     //     for (int q = 0; q < commSize; q++)
      //     //       printf("With counter: Proc[%d] steal_sizes[%d] = %d\n", MPIRank, q, steal_sizes[q]);
      //     //     printf("ERROR: I Proc[%d] never receive work\n", MPIRank);
      //     //     // exit(-1);
      //     //   }
      //     //   // continue;
      //     // }
      //     // time2 = omp_get_wtime();

      //     // sync_sizes += time2 - time1;

      //     printf("After While: Proc[%d] got [%d] work from StolenProc[%d], counter = %d\n", MPIRank, steal_sizes[responder], responder, counter);
      //     pushBackBulk(&pool_lloc, (steal_nodes + (responder * capacity_per_proc)), steal_sizes[responder]);
      //     steal_sizes[responder] = 0;
      //     steal_request[MPIRank] = -1;
      //     for (int k = 0; k < commSize; k++)
      //     {
      //       if (k == MPIRank)
      //         continue; // Skip self
      //                   // MPI_Win_lock(MPI_LOCK_EXCLUSIVE, k, 0, win_requests);
      //                   // MPI_Put(&steal_request[MPIRank], 1, MPI_INT, k, MPIRank, 1, MPI_INT, win_requests);
      //                   // // MPI_Win_flush(k, win_requests);
      //                   // MPI_Win_flush_all(win_requests);
      //                   // MPI_Win_sync(win_requests);
      //                   // MPI_Win_unlock(k, win_requests);

      //       // MPI_Win_sync(win_requests);
      //       //  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, k, 0, win_sizes);
      //       //  MPI_Put(&steal_sizes[responder], 1, MPI_INT, k, responder, 1, MPI_INT, win_sizes);
      //       //  MPI_Win_flush(k, win_sizes);
      //       //  MPI_Win_unlock(k, win_sizes);
      //       //  MPI_Win_sync(win_sizes);
      //       //  MPI_Win_sync(win_sizes);
      //       //  MPI_Win_sync(win_requests);
      //       //  MPI_Win_sync(win_nodes);
      //     }
      //   }
      //   if (steal_request[MPIRank] == -1)
      //     break;
      //   if (sum == commSize * (-2)) // Everybody has no more work
      //   {
      //     // In this implementation, if remoteSteal == false, then we are probably over
      //     remoteSteal = false;
      //     break;
      //   }
      //   // else
      //   //   break;
      // }

      // // Distributed Termination Condition
      // // This termination condition is not 'ideal' for distributed level
      // if (remoteSteal == false)
      // {
      //   printf("Proc[%d] Are we all done ?\n", MPIRank);
      //   if (atomic_load(&localeState) == BUSY)
      //     atomic_store(&localeState, IDLE);

      //   bool *allLocaleStateTemp = (bool *)malloc(commSize * sizeof(bool));
      //   _Atomic bool *allLocaleState = (_Atomic bool *)malloc(commSize * sizeof(_Atomic bool));
      //   bool eachLocaleStateTemp = atomic_load(&localeState);

      //   // Gather boolean states from all processes to all processes
      //   MPI_Allgather(&eachLocaleStateTemp, 1, MPI_C_BOOL, allLocaleStateTemp, 1, MPI_C_BOOL, MPI_COMM_WORLD);
      //   for (int i = 0; i < commSize; i++)
      //     atomic_store(&allLocaleState[i], allLocaleStateTemp[i]);

      //   // Check if eachLocalState of every process agrees for termination
      //   if (allIdle(allLocaleState, commSize, &allLocalesIdleFlag))
      //   {
      //     free(allLocaleStateTemp);
      //     free(allLocaleState);
      //     // time2Idle = omp_get_wtime();
      //     // IdleTime += time2Idle - time1Idle;
      //     break;
      //   }
      //   // time2Idle = omp_get_wtime();
      //   // IdleTime += time2Idle - time1Idle;
      //   continue;
      // }
      // else
      // {
      //   // time2Idle = omp_get_wtime();
      //   // IdleTime += time2Idle - time1Idle;
      //   continue;
      // }
    }
  }

  printf("MPIRank[%d] is after GPU-accelerated while\n", MPIRank);

  for (int i = 0; i < commSize; i++)
    printf("Proc[%d] steal_request[%d] = %d\n", MPIRank, i, steal_request[i]);

  printf("Proc[%d] nSSteal = %d nSteal = %d pool_lloc.size = %d Stolen = [%d]\n", MPIRank, nSSteal, nSteal, pool_lloc.size, stolen);
  // printf("Proc[%d] put_sizes = %.4f, put_request = %.4f, put_nodes = %.4f, compare_request = %.4f, sync_sizes = %.4f\n GPUWork = %.4f, cpyBounds = %.4f, cpyKernel = %.4f, cpyParents = %.4f, GPUTime = %.4f, IdleTime = %.4f\n",
  //  MPIRank, put_sizes, put_request, put_nodes, compare_request, sync_sizes, GPUWork, cpyBounds, cpyKernel, cpyParents, GPUTime, IdleTime);

  MPI_Barrier(MPI_COMM_WORLD);

  // Freeing device and host memory from GPU-accelerated step (step 2)
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
  free(children);
  free(stolenNodes);
  free(bounds);
  free(sumOffSets);
  free(nodeIndex);

  /*******************************
  Gathering statistics
  *******************************/
  endTime = omp_get_wtime();
  double t2, t2Temp = endTime - startTime;
  // double maxDevice = get_max(timeDevice, D);
  // t2Temp -= maxDevice;
  MPI_Reduce(&t2Temp, &t2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (MPIRank == 0) // First BFS done on all MPI processes, values accounted only on MPI 0
  {
    eachLocaleExploredTree += *exploredTree;
    eachLocaleExploredSol += *exploredSol;
  }
  MPI_Reduce(&eachLocaleExploredTree, exploredTree, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // At this point, value per process
  MPI_Reduce(&eachLocaleExploredSol, exploredSol, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);   // At this point, value per process
  MPI_Allreduce(&eachLocaleBest, best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);                                // Take minimum between all best known upper-bounds

  MPI_Gather(&eachLocaleExploredTree, 1, MPI_UNSIGNED_LONG_LONG, all_expTreeGPU, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(&eachLocaleExploredSol, 1, MPI_UNSIGNED_LONG_LONG, all_expSolGPU, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  // MPI_Gather(genChildGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_genChildGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  // MPI_Gather(nbStealsGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_nbStealsGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  // MPI_Gather(nbSStealsGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_nbSStealsGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  // MPI_Gather(nbTerminationGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_nbTerminationGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  // MPI_Gather(&nbSLocDistLoadBal, 1, MPI_UNSIGNED_LONG_LONG, nbSDistLoadBal, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // MPI_Gather(timeGpuCpy, NB_THREADS_MAX, MPI_DOUBLE, all_timeGpuCpy, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // MPI_Gather(timeGpuMalloc, NB_THREADS_MAX, MPI_DOUBLE, all_timeGpuMalloc, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // MPI_Gather(timeGpuKer, NB_THREADS_MAX, MPI_DOUBLE, all_timeGpuKer, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // MPI_Gather(timeGenChild, NB_THREADS_MAX, MPI_DOUBLE, all_timeGenChild, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // MPI_Gather(timePoolOps, NB_THREADS_MAX, MPI_DOUBLE, all_timePoolOps, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // MPI_Gather(timeGpuIdle, NB_THREADS_MAX, MPI_DOUBLE, all_timeGpuIdle, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // MPI_Gather(timeTermination, NB_THREADS_MAX, MPI_DOUBLE, all_timeTermination, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // MPI_Gather(&timeLocLoadBal, 1, MPI_DOUBLE, timeLoadBal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Print info
  if (MPIRank == 0)
  {
    printf("\nSearch on GPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("  Per MPI process: ");
    for (int i = 0; i < commSize; i++)
      printf("[%d] = %llu ", i, all_expTreeGPU[i]);
    printf("\n");
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("  Per MPI process: ");
    for (int i = 0; i < commSize; i++)
      printf("[%d] = %llu ", i, all_expSolGPU[i]);
    printf("\n");
    printf("Best solution found: %d\n", *best);
    printf("Elapsed time: %f [s]\n\n", t2);
  }

  /*
    Step 3: We complete the depth-first search on CPU.
  */
  unsigned long long int final_expTree = 0, final_expSol = 0, all_final_expTree = 0, all_final_expSol = 0;
  startTime = omp_get_wtime();
  while (1)
  {
    int hasWork = 0;
    Node parent = popBackFree(&pool_lloc, &hasWork);
    if (!hasWork)
      break;
    decompose(jobs, lb, &eachLocaleBest, lbound1, lbound2, parent, &final_expTree, &final_expSol, &pool_lloc);
  }
  endTime = omp_get_wtime();
  double t3, t3Temp = endTime - startTime;
  MPI_Reduce(&t3Temp, &t3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&final_expTree, &all_final_expTree, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&final_expSol, &all_final_expSol, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&eachLocaleBest, best, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  *elapsedTime = t1 + t2 + t3;

  // Freeing memory for structs
  deleteSinglePool_atom(&pool);
  deleteSinglePool_atom(&pool_lloc);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);

  if (MPIRank == 0)
  {
    *exploredTree += all_final_expTree;
    *exploredSol += all_final_expSol;

    printf("\nSearch on CPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("Elapsed time: %f [s]\n", t3);
    printf("\nExploration terminated.\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  // MPI_Win_free(&win_nodes);
  MPI_Win_free(&win_requests);
  MPI_Win_free(&win_sizes);
  MPI_Type_free(&myNode);
}

int main(int argc, char *argv[])
{
  int provided, MPIRank, commSize;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  if (provided != MPI_THREAD_SINGLE)
  {
    printf("MPI does not support multi-threaded implementation.\n");
    MPI_Finalize();
    return -1;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRank);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);

  // ERROR HANDLING
  if (commSize <= 0)
  {
    fprintf(stderr, "Error: commSize (%d) must be greater than zero.\n", commSize);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int flag;
  MPI_Initialized(&flag);
  if (!flag)
  {
    fprintf(stderr, "Error: MPI is not initialized.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  srand(time(NULL));

  int version = 2; // MPI Multi-GPU version is code 2
  // MPI Multi-GPU PFSP only uses: inst, lb, ub, m, M, commsize, LB
  int inst, lb, ub, m, M, T, D, C, ws, LB;
  double perc;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &T, &D, &C, &ws, &LB, &perc);

  // Here it can go a future step providing multi-core option
  // For now, only multi-GPU will be taken into account (Variable commSize)

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  if (MPIRank == 0)
    print_settings(inst, machines, jobs, ub, lb, commSize, C, ws, D, LB, version); // OBS: Here commSize and D switch places

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0, exploredSol = 0;
  unsigned long long int *all_expTreeGPU, *all_expSolGPU, *all_genChildGPU, *all_nbStealsGPU, *all_nbSStealsGPU, *all_nbTerminationGPU, nbSDistLoadBal[commSize];

  double elapsedTime = 0;
  double *all_timeGpuCpy, *all_timeGpuMalloc, *all_timeGpuKer, *all_timeGenChild, *all_timePoolOps, *all_timeGpuIdle, *all_timeTermination, timeLoadBal[commSize];
  int NB_THREADS_MAX = commSize; // For now, the amount of MPI processes

  for (int i = 0; i < commSize; i++)
  {
    nbSDistLoadBal[i] = 0;
    timeLoadBal[i] = 0;
  }
  if (MPIRank == 0)
  {
    all_expTreeGPU = (unsigned long long int *)malloc(commSize * NB_THREADS_MAX * sizeof(unsigned long long int));
    all_expSolGPU = (unsigned long long int *)malloc(commSize * NB_THREADS_MAX * sizeof(unsigned long long int));
    all_genChildGPU = (unsigned long long int *)malloc(commSize * NB_THREADS_MAX * sizeof(unsigned long long int));
    all_nbStealsGPU = (unsigned long long int *)malloc(commSize * NB_THREADS_MAX * sizeof(unsigned long long int));
    all_nbSStealsGPU = (unsigned long long int *)malloc(commSize * NB_THREADS_MAX * sizeof(unsigned long long int));
    all_nbTerminationGPU = (unsigned long long int *)malloc(commSize * NB_THREADS_MAX * sizeof(unsigned long long int));
    all_timeGpuCpy = (double *)malloc(commSize * NB_THREADS_MAX * sizeof(double));
    all_timeGpuMalloc = (double *)malloc(commSize * NB_THREADS_MAX * sizeof(double));
    all_timeGpuKer = (double *)malloc(commSize * NB_THREADS_MAX * sizeof(double));
    all_timeGenChild = (double *)malloc(commSize * NB_THREADS_MAX * sizeof(double));
    all_timePoolOps = (double *)malloc(commSize * NB_THREADS_MAX * sizeof(double));
    all_timeGpuIdle = (double *)malloc(commSize * NB_THREADS_MAX * sizeof(double));
    all_timeTermination = (double *)malloc(commSize * NB_THREADS_MAX * sizeof(double));
  }

  pfsp_search(inst, lb, m, M, T, D, C, LB, MPIRank, commSize, perc, &optimum, &exploredTree, &exploredSol, &elapsedTime,
              all_expTreeGPU, all_expSolGPU, all_genChildGPU, all_nbStealsGPU, all_nbSStealsGPU, all_nbTerminationGPU, nbSDistLoadBal,
              all_timeGpuCpy, all_timeGpuMalloc, all_timeGpuKer, all_timeGenChild, all_timePoolOps, all_timeGpuIdle, all_timeTermination, timeLoadBal);

  if (MPIRank == 0)
  {
    print_results(optimum, exploredTree, exploredSol, elapsedTime);
    print_results_file_dist_multi_gpu(inst, lb, D, C, LB, commSize, optimum, m, M, T, exploredTree, exploredSol, elapsedTime,
                                      all_expTreeGPU, all_expSolGPU, all_genChildGPU, all_nbStealsGPU, all_nbSStealsGPU, all_nbTerminationGPU, nbSDistLoadBal,
                                      all_timeGpuCpy, all_timeGpuMalloc, all_timeGpuKer, all_timeGenChild, all_timePoolOps, all_timeGpuIdle, all_timeTermination, timeLoadBal);
  }
  if (MPIRank == 0)
  {
    free(all_expTreeGPU);
    all_expTreeGPU = NULL;
    free(all_expSolGPU);
    all_expSolGPU = NULL;
    free(all_genChildGPU);
    all_genChildGPU = NULL;
    free(all_nbStealsGPU);
    all_nbStealsGPU = NULL;
    free(all_nbSStealsGPU);
    all_nbSStealsGPU = NULL;
    free(all_nbTerminationGPU);
    all_nbTerminationGPU = NULL;
    free(all_timeGpuCpy);
    all_timeGpuCpy = NULL;
    free(all_timeGpuMalloc);
    all_timeGpuMalloc = NULL;
    free(all_timeGpuKer);
    all_timeGpuKer = NULL;
    free(all_timeGenChild);
    all_timeGenChild = NULL;
    free(all_timePoolOps);
    all_timePoolOps = NULL;
    free(all_timeGpuIdle);
    all_timeGpuIdle = NULL;
    free(all_timeTermination);
    all_timeTermination = NULL;
  }
  MPI_Finalize();
  return 0;
}