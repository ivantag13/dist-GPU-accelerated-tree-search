/*
Distributed multi-node multi-core multi-GPU B&B to solve Taillard instances of the PFSP based on MPI+OpenMP+CUDA written in C language
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

int globalTermination(int commSize, int D, SinglePool_atom *multiPool, int *poolSizes_all, int m)
{
  int termination_flag = 0;
  int global_flags[commSize];
  for (int i = 0; i < D; i++)
  {
    if (multiPool[i].size >= m || poolSizes_all[i] >= m)
      termination_flag = 1; // If still work available
  }
  MPI_Allgather(&termination_flag, 1, MPI_INT, global_flags, 1, MPI_INT, MPI_COMM_WORLD);
  termination_flag = 0;
  for (int i = 0; i < commSize; i++)
    termination_flag += global_flags[i];

  // TODO: poolSizes_all has to be rechecked? No, because once recovered this value is not set to zero after work done on GPU
  if (termination_flag == 0)
    return 1;
  else
    return 0;
}

// TODO: Experimental Termination condition
// int globalTermination(int commSize, _Atomic bool *allTasksIdleFlag, _Atomic bool *eachTaskTermination, int NB_THREADS_COMPUTE, int MPIRank)
// {
//   int local_flag = 0;
//   int global_flag = 0;
//   int sum = 0;

//   if (atomic_load(allTasksIdleFlag))
//   {
//     while (sum < NB_THREADS_COMPUTE)
//     {
//       sum = 0;
//       for (int i = 0; i < NB_THREADS_COMPUTE; i++)
//       {
//         if (atomic_load(&eachTaskTermination[i]))
//           sum++;
//       }
//     }
//     local_flag = 1;
//   }
//   else
//     local_flag = 0;

//   MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
//   printf("Proc[%d] global_flag = %d\n", MPIRank, global_flag);

//   if (global_flag == commSize)
//     return 1;
//   else
//     return 0;
// }

int gatherRecvDataV(int commSize, int sendSize, Node *sendData, Node *recvData, MPI_Datatype myData)
{
  int sendCounts[commSize];
  int recvCounts[commSize];
  int recvDispls[commSize];
  int recvNodesSize = 0;
  MPI_Allgather(&sendSize, 1, MPI_INT, sendCounts, 1, MPI_INT, MPI_COMM_WORLD);
  for (int i = 0; i < commSize; i++)
  {
    recvCounts[i] = sendCounts[i];
    recvDispls[i] = recvNodesSize;
    recvNodesSize += recvCounts[i];
  }
  MPI_Allgatherv(sendData, sendSize, myData, recvData, recvCounts, recvDispls, myData, MPI_COMM_WORLD);
  return recvNodesSize;
}

/***********************************************************************************
Implementation of the parallel Distributed Multi-GPU C+MPI+OpenMP+CUDA PFSP search.
***********************************************************************************/
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

  // TODO: if no multi-core is activated (C==0) one need to fix how the mapping should be fixed in case of inter-node work stealing
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  int nb_proc;
  if (C == 1)
    nb_proc = omp_get_num_procs();
  else if (C == 0)
    nb_proc = deviceCount;
  int NB_THREADS_GPU = (nb_proc / deviceCount);
  int NB_THREADS_MAX = D * NB_THREADS_GPU;

  // printf("Proc[%d] Num_procs[%d] MAX_GPU[%d] NB_THREADS_GPU[%d] NB_THREADS_MAX[%d]\n", MPIRank, nb_procs, MAX_GPU, NB_THREADS_GPU, NB_THREADS_MAX);

  // Initialize problem parameters
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  // Initialize pool
  Node root;
  initRoot(&root, jobs);
  SinglePool_atom pool;
  initSinglePool_atom(&pool);
  pushBack(&pool, root);

  // Global timer
  double startTime, endTime;
  startTime = omp_get_wtime();

  // Bounding data (constant data)
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

  while (pool.size < commSize * NB_THREADS_MAX * m)
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

  int NB_THREADS_COMPUTE = NB_THREADS_MAX - 1;
  SinglePool_atom multiPool[NB_THREADS_COMPUTE];
  for (int i = 0; i < NB_THREADS_COMPUTE; i++)
    initSinglePool_atom(&multiPool[i]);
  unsigned long long int expTreeGPU[NB_THREADS_MAX], expSolGPU[NB_THREADS_MAX], genChildGPU[NB_THREADS_MAX], nbStealsGPU[NB_THREADS_MAX], nbSStealsGPU[NB_THREADS_MAX], nbTerminationGPU[NB_THREADS_MAX], nbSLocDistLoadBal = 0;
  double timeGpuCpy[NB_THREADS_MAX], timeGpuMalloc[NB_THREADS_MAX], timeGpuKer[NB_THREADS_MAX], timeGenChild[NB_THREADS_MAX], timePoolOps[NB_THREADS_MAX], timeGpuIdle[NB_THREADS_MAX], timeTermination[NB_THREADS_MAX], timeLocLoadBal = 0, timeDevice[NB_THREADS_MAX];
  int best_l = *best;
  unsigned long long int tree = 0, sol = 0;

  for (int i = 0; i < NB_THREADS_MAX; i++)
  {
    expTreeGPU[i] = 0;
    expSolGPU[i] = 0;
    genChildGPU[i] = 0;
    nbStealsGPU[i] = 0;
    nbSStealsGPU[i] = 0;
    nbTerminationGPU[i] = 0;
    timeGpuCpy[i] = 0;
    timeGpuMalloc[i] = 0;
    timeGpuKer[i] = 0;
    timeGenChild[i] = 0;
    timePoolOps[i] = 0;
    timeGpuIdle[i] = 0;
    timeTermination[i] = 0;
    timeDevice[i] = 0;
  }

  // Atomic boolean variables
  _Atomic bool bestLock = false;
  _Atomic bool allTasksIdleFlag = false;
  _Atomic bool eachTaskState[NB_THREADS_COMPUTE]; // one task per GPU
  _Atomic bool eachTaskTermination[NB_THREADS_COMPUTE];
  for (int i = 0; i < NB_THREADS_COMPUTE; i++)
  {
    atomic_store(&eachTaskState[i], BUSY);
    atomic_store(&eachTaskTermination[i], BUSY);
  }
  int global_termination_flag = 0, local_need = 0;
  int poolSizes_all[NB_THREADS_COMPUTE];

  startTime = omp_get_wtime();

  // TODO: This needs to be fixed when not multi-core activated
  int nbThreads = (L != 0) ? NB_THREADS_MAX : NB_THREADS_COMPUTE;

  // Allocating vectors for distributed communications
  int distRatio = 3;
  Node *sendNodes = (Node *)malloc(distRatio * M * sizeof(Node));
  Node *recvNodes = (Node *)malloc(commSize * distRatio * M * sizeof(Node));
  Node *insNodes = (Node *)malloc(commSize * distRatio * M * sizeof(Node));

#pragma omp parallel num_threads(nbThreads) shared(bestLock, eachTaskState, allTasksIdleFlag, pool_lloc, multiPool, jobs, machines, lbound1, lbound2,      \
                                                       lb, m, M, D, perc, L, best, exploredTree, exploredSol, MPIRank, commSize, global_termination_flag,  \
                                                       poolSizes_all, sendNodes, recvNodes, insNodes, elapsedTime, expTreeGPU, expSolGPU, genChildGPU,     \
                                                       nbStealsGPU, nbSStealsGPU, nbTerminationGPU, nbSDistLoadBal, timeGpuCpy, timeGpuMalloc, timeGpuKer, \
                                                       timeGenChild, timePoolOps, timeGpuIdle, timeTermination, timeLoadBal, timeDevice)                   \
    reduction(min : best_l) reduction(+ : tree, sol)
  {
    double startGpuCpy, endGpuCpy, startGpuMalloc, endGpuMalloc, startGpuKer, endGpuKer, startGenChild, endGenChild,
        startPoolOps, endPoolOps, startGpuIdle, endGpuIdle, startTermination, endTermination, startLoadBal, endLoadBal;
    int cpuID = omp_get_thread_num(), nbSteals = 0, nbSSteals = 0;
    int cpulb = lb;
    if (lb == 1)
      cpulb = 0;
    // Debug: printf("Debug: Proc[%d] Thread[%d] Mark[%d]\n", MPIRank, cpuID, mark); // TODO: Check like JP was doing it

    // if (D > 0 && cpuID < D && cpuID != NB_THREADS_COMPUTE)
    //   cudaSetDevice(cpuID);
    if (cpuID % NB_THREADS_GPU == 0)
      cudaSetDevice(cpuID / NB_THREADS_GPU);

    SinglePool_atom *pool_loc;
    if (cpuID != NB_THREADS_COMPUTE)
      pool_loc = &multiPool[cpuID];
    SinglePool_atom parentsPool, childrenPool;
    initSinglePool_atom(&parentsPool);
    initSinglePool_atom(&childrenPool);
    bool taskState = BUSY;

    // Each shared memory pool gets its chunk
    if (cpuID != NB_THREADS_COMPUTE)
      roundRobin_distribution(pool_loc, &pool_lloc, cpuID, NB_THREADS_COMPUTE);
#pragma omp barrier
    pool_lloc.front = 0;
    pool_lloc.size = 0;

    startGpuMalloc = omp_get_wtime();
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
    // cudaMalloc((void **)&parents_d, M * sizeof(Node));
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

#pragma omp barrier

    int counter = 0;
    while (1)
    {
      counter++;
      // Check Distributed Termination Flag
      if (global_termination_flag && L != 0)
        break;

      // Work Stealing by Last Thread
      if (cpuID == NB_THREADS_COMPUTE && L == 1)
      {
        startLoadBal = omp_get_wtime();

        // Distributed sharing of runtime discovered upper-bound
        if (best_l != *best)
          checkBest(&best_l, best, &bestLock);
        int best_all;
        MPI_Allreduce(&best_l, &best_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (best_all < *best)
          checkBest(&best_all, best, &bestLock);

        global_termination_flag = globalTermination(commSize, NB_THREADS_COMPUTE, multiPool, poolSizes_all, m);
        // global_termination_flag = globalTermination(commSize, &allTasksIdleFlag, eachTaskTermination, NB_THREADS_COMPUTE, MPIRank);

        // No global termination, then work stealing
        if (!global_termination_flag)
        {
          // Step 1: Share info on needy MPI processes
          int needs_work = local_need;
          int all_needs_work[commSize];
          MPI_Allgather(&needs_work, 1, MPI_INT, all_needs_work, 1, MPI_INT, MPI_COMM_WORLD);
          int needy_count = 0; // Count how many processes need work
          for (int i = 0; i < commSize; i++)
          {
            if (all_needs_work[i])
              needy_count++;
          }

          // Step 2: Determine how much to give to steal request (if this process has work)
          int sendNodesSize = 0;
          if (needy_count > 0 && needy_count < commSize && !needs_work) //  Only proceed if some (but not all) processes need work
          {
            int victims[D];
            permute(victims, D);
            for (int j = 0; j < D; j++)
            {
              int gpuVictim = victims[j] * NB_THREADS_GPU;
              sendNodesSize = popBackBulk(&multiPool[gpuVictim], m, distRatio * M, sendNodes, 2);
              if (sendNodesSize > 0)
                break;
            }
          }

          // Step 2 (optional for pure multi-core version): Determine how much to give to steal request (if this process has work)
          if (D == 0)
          {
            int sendNodesSize = 0;
            if (needy_count > 0 && needy_count < commSize && !needs_work) //  Only proceed if some (but not all) processes need work
            {
              int victims[C];
              permute(victims, C);
              for (int j = 0; j < C; j++)
              {
                int cpuVictim = victims[j] + D;
                sendNodesSize = popBackBulk(&multiPool[cpuVictim], m, distRatio * T, sendNodes, 2);
                if (sendNodesSize > 0)
                  break;
              }
            }
          }

          // DEBUG: // if (counter % 100 == 0)  //   printf("Proc[%d] Counter[%d]\n", MPIRank, counter);

          // Step 3: Gather the sizes of the send data, compute their displacements, and send nodes to recvNodes buffer
          int recvNodesSize = gatherRecvDataV(commSize, sendNodesSize, sendNodes, recvNodes, myNode);

          // Step 4: Redistribute nodes (only for processes that needed work)
          if (needs_work && recvNodesSize > 0)
          {
            nbSLocDistLoadBal++;
            int nodesPerProcess = recvNodesSize / needy_count; // Number of nodes each needy process will recover from every other process
            int remainder = recvNodesSize % needy_count;       // Remainder to handle uneven distribution to all needy MPI processes
            int insNodesSize;
            int needy_position = 0;
            for (int i = 0; i < MPIRank; i++) // Compute position in the list of needy processes
            {
              if (all_needs_work[i])
                needy_position++;
            }
            for (insNodesSize = 0; insNodesSize < nodesPerProcess; insNodesSize++)
              insNodes[insNodesSize] = recvNodes[insNodesSize * needy_count + needy_position];
            if (remainder > 0 && needy_position < remainder) // Remainder per each process (if any)
              insNodes[insNodesSize++] = recvNodes[nodesPerProcess * needy_count + needy_position];

            // TODO: improve insertion of stolen nodes into multiple local pools

            int nodesPerPool = insNodesSize / D;
            int remainderPool = insNodesSize % D;
            for (int k = 0; k < D - 1; k++)
            {
              pushBackBulk(&multiPool[k * NB_THREADS_GPU], insNodes + (nodesPerPool * k), nodesPerPool);
              atomic_store(&eachTaskState[k * NB_THREADS_GPU], BUSY);
            }
            pushBackBulk(&multiPool[(D - 1) * NB_THREADS_GPU], insNodes + (nodesPerPool * (D - 1)), nodesPerPool + remainderPool);
            atomic_store(&eachTaskState[(D - 1) * NB_THREADS_GPU], BUSY);

            // pushBackBulk(&multiPool[0], insNodes, insNodesSize);
            // atomic_store(&eachTaskState[0], BUSY);
            local_need = 0;
            atomic_store(&allTasksIdleFlag, BUSY);
          }
        }
        endLoadBal = omp_get_wtime();
        timeLocLoadBal += endLoadBal - startLoadBal;
      }

      if (cpuID != NB_THREADS_COMPUTE)
      {
        // Each task gets its parents nodes from the pool
        // counter++;

        int poolSize;
        startPoolOps = omp_get_wtime();
        if (cpuID % NB_THREADS_GPU != 0)
          poolSize = popBackBulk(pool_loc, m, T, parents, 1);
        else
          poolSize = popBackBulk(pool_loc, m, M, parents, 1);
        poolSizes_all[cpuID] = poolSize;
        endPoolOps = omp_get_wtime();
        timePoolOps[cpuID] += endPoolOps - startPoolOps;

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
            expTreeGPU[cpuID] = tree;
            expSolGPU[cpuID] = sol;

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

            const int numBounds = sum;
            const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);
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
          }
        }
        else // local work stealing
        {
          startGpuIdle = omp_get_wtime();
          int tries = 0;
          bool steal = false;
          int victims[NB_THREADS_COMPUTE];
          permute(victims, NB_THREADS_COMPUTE);
          bool expected;

          while (tries < NB_THREADS_COMPUTE && steal == false)
          { // WS0 loop
            const int victimID = victims[tries];

            if (victimID != cpuID)
            { // if not me
              SinglePool_atom *victim;
              victim = &multiPool[victimID];
              nbSteals++;
              int nn = 0;
              // int count = 0;
              while (nn < 10)
              { // WS1 loop
                expected = false;
                // count++;
                if (atomic_compare_exchange_strong(&(victim->lock), &expected, true))
                { // get the lock
                  int size = victim->size;
                  if (size >= 2 * m)
                  {
                    // int stolenNodesSize = popBackBulkFree(victim, m, 5 * M, stolenNodes, 2);
                    int stolenNodesSize;
                    if (cpuID % NB_THREADS_GPU == 0)
                      stolenNodesSize = popBackBulkFree(victim, m, 5 * M, stolenNodes, 2); // ratio is 2
                    else
                      stolenNodesSize = popBackBulkFree(victim, m, 4 * T, stolenNodes, 2); // ratio is 2

                    if (stolenNodesSize == 0)
                    {                                       // safety check
                      atomic_store(&(victim->lock), false); // reset lock
                      printf("\nProc[%d] Thread[%d] DEADCODE\n", MPIRank, cpuID);
                      exit(-1);
                    }

                    startPoolOps = omp_get_wtime();
                    pushBackBulk(pool_loc, stolenNodes, stolenNodesSize);
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
          if (steal == false)
          {
            // termination
            startTermination = omp_get_wtime();
            nbTerminationGPU[cpuID]++;
            if (taskState == BUSY)
            {
              taskState = IDLE;
              atomic_store(&eachTaskState[cpuID], IDLE);
            }
            if (allIdle(eachTaskState, NB_THREADS_COMPUTE, &allTasksIdleFlag))
            {
              if (L == 1)
              {
                // Set request for work while checking global termination
                atomic_store(&eachTaskTermination[cpuID], IDLE);
                local_need = 1;
                // atomic_store(&(pool_loc->lock), false);
                double cpuIdleStart = omp_get_wtime();
                double lastPrintTime = cpuIdleStart;
                while (atomic_load(&allTasksIdleFlag))
                {
                  double now = omp_get_wtime();
                  if (now - lastPrintTime >= 10.0)
                  {
                    printf("[%.2f] Proc[%d] Thread[%d] Still Idle\n", now - cpuIdleStart, MPIRank, cpuID);
                    lastPrintTime = now;
                  }
                  usleep(50);
                  if (global_termination_flag)
                  {
                    // printf("Proc[%d] Thread[%d] Checked Global Termination Condition\n", MPIRank, cpuID);
                    break;
                  }
                }
                atomic_store(&eachTaskTermination[cpuID], BUSY);
                // continue;
              }
              else
              {
                endTermination = omp_get_wtime();
                timeTermination[cpuID] += endTermination - startTermination;
                break;
              }
            }
            endTermination = omp_get_wtime();
            timeTermination[cpuID] += endTermination - startTermination;
            continue;
          }
          // // TODO: Second experimental termination condition
          // if (localSteal == false && globalSteal == false)
          //         {
          //           // Intra-node termination detection
          //           if (taskState == BUSY)
          //           {
          //             taskState = IDLE;
          //             atomic_store(&eachTaskState[gpuID], IDLE);
          //           }
          //           if (allIdle(eachTaskState, D, &allTasksIdleFlag))
          //           {
          //             // Inter-node termination detection
          //             if (locState == BUSY)
          //             {
          //               locState = IDLE;
          //               // All OpenMP threads access and agree on the value of the next variable at this stage
          //               atomic_store(&eachLocaleState, IDLE);
          //             }

          // // A random OpenMP thread checks the global termination condition in every process
          // #pragma omp single
          //           {
          //             bool *allLocaleStateTemp = (bool *)malloc(commSize * sizeof(bool));
          //             _Atomic bool *allLocaleState = (_Atomic bool *)malloc(commSize * sizeof(_Atomic bool));
          //             bool eachLocaleStateTemp = atomic_load(&eachLocaleState);

          //             // Gather boolean states from all processes to all processes
          //             MPI_Allgather(&eachLocaleStateTemp, 1, MPI_C_BOOL, allLocaleStateTemp, 1, MPI_C_BOOL, MPI_COMM_WORLD);
          //             for (int i = 0; i < commSize; i++)
          //               atomic_store(&allLocaleState[i], allLocaleStateTemp[i]);

          //             // Check if eachLocalState of every process agrees for termination
          //             if (allIdle(allLocaleState, commSize, &allLocalesIdleFlag))
          //               atomic_store(&globalTerminationFlag, true);

          //             free(allLocaleStateTemp);
          //             free(allLocaleState);
          //           }
          //           if (atomic_load(&globalTerminationFlag))
          //           {
          //             printf("From process %d thread %d nGSSteal = %d\n", MPIRank, omp_get_thread_num(), nGSSteal);
          //             break;
          //           }
          //         }
          //         continue;
          //       }
          else
          {
            continue;
          }
        }
      }
    }

#pragma omp barrier

    // OpenMP environment freeing variables
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

#pragma omp critical
    {
      if (cpuID != NB_THREADS_COMPUTE)
      {
        // printf("Proc[%d] Thread[%d] best_l[%d] best[%d]\n", MPIRank, cpuID, best_l, *best);
        nbStealsGPU[cpuID] = nbSteals;
        nbSStealsGPU[cpuID] = nbSSteals;
        expTreeGPU[cpuID] = tree;
        expSolGPU[cpuID] = sol;
        const int poolLocSize = pool_loc->size;
        for (int i = 0; i < poolLocSize; i++)
        {
          int hasWork = 0;
          pushBackFree(&pool_lloc, popBackFree(pool_loc, &hasWork));
          if (!hasWork)
            break;
        }
        deleteSinglePool_atom(pool_loc);
      }
    }
  } // End of parallel region OpenMP

  printf("Proc[%d] Checked Global Termination Condition\n", MPIRank);

  free(sendNodes);
  free(recvNodes);
  free(insNodes);
  MPI_Barrier(MPI_COMM_WORLD);

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
    tree += *exploredTree;
    sol += *exploredSol;
  }
  MPI_Reduce(&tree, exploredTree, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // At this point, value per process
  MPI_Reduce(&sol, exploredSol, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);   // At this point, value per process
  *best = best_l;                                                                         // Best known upper-bound within each MPI process
  MPI_Allreduce(&best_l, best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);                      // Take minimum between all between best known upper-bounds
  best_l = *best;                                                                         // Best known upper bound among all MPI processes
  // printf("Proc[%d] best_l[%d] best[%d]\n", MPIRank, best_l, *best);

  MPI_Gather(expTreeGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_expTreeGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(expSolGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_expSolGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(genChildGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_genChildGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(nbStealsGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_nbStealsGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(nbSStealsGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_nbSStealsGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(nbTerminationGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, all_nbTerminationGPU, NB_THREADS_MAX, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(&nbSLocDistLoadBal, 1, MPI_UNSIGNED_LONG_LONG, nbSDistLoadBal, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  MPI_Gather(timeGpuCpy, NB_THREADS_MAX, MPI_DOUBLE, all_timeGpuCpy, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(timeGpuMalloc, NB_THREADS_MAX, MPI_DOUBLE, all_timeGpuMalloc, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(timeGpuKer, NB_THREADS_MAX, MPI_DOUBLE, all_timeGpuKer, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(timeGenChild, NB_THREADS_MAX, MPI_DOUBLE, all_timeGenChild, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(timePoolOps, NB_THREADS_MAX, MPI_DOUBLE, all_timePoolOps, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(timeGpuIdle, NB_THREADS_MAX, MPI_DOUBLE, all_timeGpuIdle, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(timeTermination, NB_THREADS_MAX, MPI_DOUBLE, all_timeTermination, NB_THREADS_MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&timeLocLoadBal, 1, MPI_DOUBLE, timeLoadBal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Print info
  if (MPIRank == 0)
  {
    unsigned long long int expTreeProc[commSize], expSolProc[commSize];
    for (int i = 0; i < commSize; i++)
    {
      expTreeProc[i] = 0;
      expSolProc[i] = 0;
      for (int j = 0; j < NB_THREADS_MAX; j++)
      {
        expTreeProc[i] += all_expTreeGPU[i * NB_THREADS_MAX + j];
        expSolProc[i] += all_expSolGPU[i * NB_THREADS_MAX + j];
      }
    }

    printf("\nSearch on GPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("  Per MPI process: ");
    for (int i = 0; i < commSize; i++)
      printf("[%d] = %llu ", i, expTreeProc[i]);
    printf("\n");
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("  Per MPI process: ");
    for (int i = 0; i < commSize; i++)
      printf("[%d] = %llu ", i, expSolProc[i]);
    printf("\n");
    printf("Best solution found: %d\n", *best);
    printf("Elapsed time: %f [s]\n\n", t2);
  }

  /*
  Step 3: Remaining nodes evaluated in DFS on CPU by each MPI process.
  */
  unsigned long long int final_expTree = 0, final_expSol = 0, all_final_expTree = 0, all_final_expSol = 0;
  startTime = omp_get_wtime();
  while (1)
  {
    int hasWork = 0;
    Node parent = popBackFree(&pool_lloc, &hasWork);
    if (!hasWork)
      break;
    decompose(jobs, lb, &best_l, lbound1, lbound2, parent, &final_expTree, &final_expSol, &pool_lloc);
  }
  endTime = omp_get_wtime();
  double t3, t3Temp = endTime - startTime;
  MPI_Reduce(&t3Temp, &t3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&final_expTree, &all_final_expTree, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&final_expSol, &all_final_expSol, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&best_l, best, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  *elapsedTime = t1 + t2 + t3;

  // freeing memory for structs common to all MPI processes
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
  MPI_Type_free(&myNode);
}

int main(int argc, char *argv[])
{
  int provided, MPIRank, commSize;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided != MPI_THREAD_SERIALIZED)
  {
    printf("MPI does not support multiple threads.\n");
    MPI_Finalize();
    return -1;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &MPIRank);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);

  srand(time(NULL));

  int version = 3; // Distributed Multi-GPU version is code 3
  // Distributed Multi-GPU PFSP only uses: inst, lb, ub, m, M, D, LB
  int inst, lb, ub, m, M, T, D, C, ws, LB;
  double perc;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &T, &D, &C, &ws, &LB, &perc);

  if (C < 0 || C > 1)
  {
    printf("C is set to %d. Invalid option for this version.\nChoose 0 to unable and 1 to enable multi-core. Mapping automatically done.\n", C);
    exit(1);
  }

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  int nb_proc;
  if (C == 1) // Activate Multi-core
    nb_proc = omp_get_num_procs();
  else if (C == 0) // Deactivate Multi-core
    nb_proc = deviceCount;
  // int MAX_GPU = 8;
  int NB_THREADS_GPU = (nb_proc / deviceCount);
  int NB_THREADS_MAX = D * NB_THREADS_GPU;
  if (D > deviceCount)
  {
    printf("Execution Terminated. More GPU devices requested than the ones available\n");
    exit(1);
  }

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  if (MPIRank == 0)
    print_settings(inst, machines, jobs, ub, lb, D, C, ws, commSize, LB, version);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0, exploredSol = 0;
  unsigned long long int *all_expTreeGPU, *all_expSolGPU, *all_genChildGPU, *all_nbStealsGPU, *all_nbSStealsGPU, *all_nbTerminationGPU, nbSDistLoadBal[commSize];

  double elapsedTime = 0;
  double *all_timeGpuCpy, *all_timeGpuMalloc, *all_timeGpuKer, *all_timeGenChild, *all_timePoolOps, *all_timeGpuIdle, *all_timeTermination, timeLoadBal[commSize];

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

    C = NB_THREADS_MAX - D; // Only CPU Processing Units. Change for statistical evaluation.
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