/*
  Distributed multi-GPU B&B to solve Taillard instances of the PFSP in C+MPI+OpenMP+CUDA.
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
#include "../common/util.h"

/******************************************************************************
Create Node MPI data type
******************************************************************************/

void create_mpi_node_type(MPI_Datatype *mpi_node_type)
{
  int blocklengths[3] = {1, 1, MAX_JOBS};
  MPI_Aint offsets[3];
  offsets[0] = offsetof(Node, depth);
  offsets[1] = offsetof(Node, limit1);
  offsets[2] = offsetof(Node, prmu);

  MPI_Datatype types[3] = {MPI_UINT8_T, MPI_INT, MPI_INT};
  MPI_Type_create_struct(3, blocklengths, offsets, types, mpi_node_type);
  MPI_Type_commit(mpi_node_type);
}

/******************************************************************************
Statistics function
******************************************************************************/
void print_results_file(const int inst, const int machines, const int jobs, const int lb, const int D, const int w, const int commSize, const int optimum,
                        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
                        unsigned long long int *expTreeProc, unsigned long long int *expSolProc, unsigned long long int *nStealsProc,
                        double *timeKernelCall, double *timeIdle, double *workloadProc)
{
  FILE *file;
  file = fopen("distMultigpu.dat", "a");
  fprintf(file, "\nProc[%d] LB[%d] GPU[%d] ta[%d] lb[%d] time[%.4f] Tree[%llu] Sol[%llu] Best[%d]\n", commSize, w, D, inst, lb, timer, exploredTree, exploredSol, optimum);
  fprintf(file, "Workload per Proc: ");
  compute_boxplot_stats(workloadProc, commSize, file);
  fprintf(file, "Max Kernel Call Times per Proc: ");
  compute_boxplot_stats(timeKernelCall, commSize, file);
  fprintf(file, "Max Idle Times per Proc: ");
  compute_boxplot_stats(timeIdle, commSize, file);
  fprintf(file, "Load Balancing per Proc: ");
  for (int i = 0; i < commSize; i++)
  {
    if (i != commSize - 1)
      fprintf(file, "%llu ", nStealsProc[i]);
    else
      fprintf(file, "%llu\n", nStealsProc[i]);
  }
  fclose(file);

  file = fopen("distMultigpu_detail.dat", "a");
  fprintf(file, "\nProc[%d] LB[%d] GPU[%d] ta%d lb%d time[%.4f] Tree[%llu] Sol[%llu] Best[%d]\n", commSize, w, D, inst, lb, timer, exploredTree, exploredSol, optimum);
  fprintf(file, "Explored Nodes per Proc: ");
  for (int i = 0; i < commSize; i++)
  {
    if (i != commSize - 1)
      fprintf(file, "%llu ", expTreeProc[i]);
    else
      fprintf(file, "%llu\n", expTreeProc[i]);
  }
  fprintf(file, "Explored Solutions per Proc: ");
  for (int i = 0; i < commSize; i++)
  {
    if (i != commSize - 1)
      fprintf(file, "%llu ", expSolProc[i]);
    else
      fprintf(file, "%llu\n", expSolProc[i]);
  }
  fprintf(file, "Time kernelCall per Proc: ");
  for (int i = 0; i < commSize; i++)
  {
    if (i != commSize - 1)
      fprintf(file, "%.4f ", timeKernelCall[i]);
    else
      fprintf(file, "%.4f\n", timeKernelCall[i]);
  }
  fprintf(file, "Time Idle per Proc: ");
  for (int i = 0; i < commSize; i++)
  {
    if (i != commSize - 1)
      fprintf(file, "%.4f ", timeIdle[i]);
    else
      fprintf(file, "%.4f\n", timeIdle[i]);
  }
  fprintf(file, "Succesful Load Balancing per Proc: ");
  for (int i = 0; i < commSize; i++)
  {
    if (i != commSize - 1)
      fprintf(file, "%llu ", nStealsProc[i]);
    else
      fprintf(file, "%llu\n", nStealsProc[i]);
  }
  fclose(file);
  return;
}

/***********************************************************************************
Implementation of the parallel Distributed Multi-GPU C+MPI+OpenMP+CUDA PFSP search.
***********************************************************************************/
void pfsp_search(const int inst, const int lb, const int m, const int M, const int D, const int w, double perc, int *best,
                 unsigned long long int *exploredTree, unsigned long long int *exploredSol, double *elapsedTime,
                 unsigned long long int *expTreeProc, unsigned long long int *expSolProc, unsigned long long int *nStealsProc,
                 double *timeKernelCall, double *timeIdle, double *workloadProc, int MPIRank, int commSize)
{
  // New MPI data type corresponding to Node
  MPI_Datatype myNode;
  create_mpi_node_type(&myNode);

  // Initializing problem
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  // Starting pool
  Node root;
  initRoot(&root, jobs);

  SinglePool_atom pool;
  initSinglePool_atom(&pool);

  pushBack(&pool, root);

  // Timer
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
  while (pool.size < commSize * D * m)
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

  // For every MPI process
  unsigned long long int eachLocaleExploredTree = 0, eachLocaleExploredSol = 0;
  int eachLocaleBest = *best;

  // For GPUs under each MPI process
  unsigned long long int eachExploredTree[D], eachExploredSol[D];
  int eachBest[D];

  SinglePool_atom pool_lloc;
  initSinglePool_atom(&pool_lloc);

  // each MPI process gets its chunk
  roundRobin_distribution(&pool_lloc, &pool, MPIRank, commSize);
  pool.front = 0;
  pool.size = 0;

  SinglePool_atom multiPool[D];
  for (int i = 0; i < D; i++)
    initSinglePool_atom(&multiPool[i]);

  // Boolean variables for termination detection
  _Atomic bool allTasksIdleFlag = false;
  _Atomic bool eachTaskState[D]; // one task per GPU
  for (int i = 0; i < D; i++)
    atomic_store(&eachTaskState[i], BUSY);

  // TODO: Implement OpenMP reduction to variables best_l, eachExploredTree, eachExploredSol
  // int best_l = *best;
  int global_termination_flag = 0, local_need = 0, request = 0;
  int poolSizes_all[D];

  double timeDevice[D];
  double timeLocalKernelCall[D];
  double timeIdleDevice[D];
  for (int i = 0; i < D; i++)
  {
    timeLocalKernelCall[i] = 0;
    timeIdleDevice[i] = 0;
  }
  startTime = omp_get_wtime();

  int nbThreads = (w != 0) ? (D + 1) : D;

#pragma omp parallel num_threads(nbThreads) shared(eachExploredTree, eachExploredSol, eachBest, eachTaskState, allTasksIdleFlag,  \
                                                       pool_lloc, multiPool, jobs, machines, lbound1, lbound2, lb, m, M, D, perc, \
                                                       best, exploredTree, exploredSol, global_termination_flag, poolSizes_all, timeDevice) // reduction(min:best_l)
  {
    double startSetDevice, endSetDevice, startKernelCall, endKernelCall, startTimeIdle, endTimeIdle;
    int nSteal = 0, nSSteal = 0;
    int gpuID = omp_get_thread_num();

    // DEBUGGING
    // printf("From Proc[%d] Thread[%d] Started MPI+Threading\n", MPIRank, gpuID);

    // WARNING: gpuID == D does not manage a GPU!!!
    if (gpuID != D)
    {
      startSetDevice = omp_get_wtime();
      cudaSetDevice(gpuID);
      endSetDevice = omp_get_wtime();
      double timeSetDevice = endSetDevice - startSetDevice;
      timeDevice[gpuID] = timeSetDevice;
    }

    unsigned long long int tree = 0, sol = 0;
    SinglePool_atom *pool_loc;
    if (gpuID != D)
      pool_loc = &multiPool[gpuID];
    int best_l = *best;
    bool taskState = BUSY;
    bool expected = false;

    // Each shared memory pool gets its chunk
    if (gpuID != D)
      roundRobin_distribution(pool_loc, &pool_lloc, gpuID, D);
#pragma omp barrier
    pool_lloc.front = 0;
    pool_lloc.size = 0;

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

    int indexChildren;

#pragma omp barrier

    int termination_flag = 1;
    int global_flags[commSize];
    int counter = 0;

    // DEBUGGING
    // printf("From Proc[%d] Thread[%d] Before While Loop\n", MPIRank, gpuID);

    while (1)
    {
      counter++;
      // Distributed Termination Flag reached, break from distributed multi-threaded environment
      if (global_termination_flag && w != 0)
      {
        // DEBUGGING
        // printf("From Proc[%d] Thread[%d] Global Termination Reached\n", MPIRank, gpuID);
        break;
      }

      // Work Sharing by Last Thread
      if (gpuID == D && w == 1)
      {
        // Termination Detection of GPU-accelerated step
        termination_flag = 0;

        // Check multiPool sizes
        for (int i = 0; i < D; i++)
        {
          if (multiPool[i].size > m || poolSizes_all[i] > 0)
            termination_flag = 1; // If still work available
        }

        // Check all locale termination flags
        MPI_Allgather(&termination_flag, 1, MPI_INT, global_flags, 1, MPI_INT, MPI_COMM_WORLD);
        int termination = 0;
        for (int i = 0; i < commSize; i++)
          termination += global_flags[i];

        // Warning: poolSizes_all has to be rechecked?
        // int sumPoolSizes = 0;
        // for (int i = 0; i < D; i++)
        //   sumPoolSizes += poolSizes_all[i];
        if (termination == 0) // || sumPoolSizes == 0)
          global_termination_flag = 1;

        // If no Termination, we proceed to work sharing/work stealing
        if (!global_termination_flag)
        {
          // Step 1 : Determine and allocate amount for shared data
          Node *sharedNodes = NULL;
          int sharedSize = 0;
          int halfSizes;
          int victims[D];
          permute(victims, D);
          for (int j = 0; j < D; j++)
          {
            Node *sharedNodesPartial;
            sharedNodesPartial = popBackBulkHalf(&multiPool[victims[j]], m, M, &halfSizes);

            // If halfSizes > 0, there is local data to share
            if (halfSizes > 0)
            {
              if (sharedNodes == NULL)
                sharedNodes = (Node *)malloc(halfSizes * sizeof(Node));
              else
                sharedNodes = (Node *)realloc(sharedNodes, (sharedSize + halfSizes) * sizeof(Node));

              // Optimization(?)
              // memcpy(sharedNodes, parents + (poolSize - halfSize), halfSize * sizeof(Node));
              for (int k = 0; k < halfSizes; k++)
                sharedNodes[sharedSize + k] = sharedNodesPartial[k];
              sharedSize += halfSizes;
              free(sharedNodesPartial);
              break;
            }
          }
          if (sharedSize > 0)
            nStealsProc[MPIRank]++;

          int sendCounts[commSize];
          int recvCounts[commSize];
          int recvDispls[commSize];

          // Step 2: Gather the sizes of the shared data (sharedSize) from all processess
          MPI_Allgather(&sharedSize, 1, MPI_INT, sendCounts, 1, MPI_INT, MPI_COMM_WORLD);

          // DEBUGGING
          //  if (counter % 100 == 0)
          //    printf("Proc[%d] sharedSize = %d at counter[%d]\n", MPIRank, sharedSize, counter);

          // Step 3: Compute displacements for the received data
          int totalReceived = 0;
          for (int i = 0; i < commSize; i++)
          {
            recvCounts[i] = sendCounts[i];
            recvDispls[i] = totalReceived;
            totalReceived += recvCounts[i];
          }

          // DEBUGGING
          // if (counter % 100 == 0)
          //   printf("Proc[%d] totalReceived = %d at counter[%d]\n", MPIRank, totalReceived, counter);

          // Step 4: Allocate buffer to store all received shared data
          Node *receivedNodes = (Node *)malloc(totalReceived * sizeof(Node));
          if (receivedNodes == NULL)
          {
            fprintf(stderr, "Proc[%d] Thread[%d] Memory allocation failed for receivedNodes\n", MPIRank, gpuID);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
          }

          // Step 5: Gather all shared data into the receivedNodes buffer
          MPI_Allgatherv(sharedNodes, sharedSize, myNode,
                         receivedNodes, recvCounts, recvDispls, myNode,
                         MPI_COMM_WORLD);

          // Step 6 : Reincorporate shared nodes into the parents vector
          int nodesPerProcess = totalReceived / commSize; // Number of nodes each process will recover from every other process
          int remainder = totalReceived % commSize;       // Remainder to handle uneven distribution

          // Nodes process per each process
          Node *insertNodes = (Node *)malloc((nodesPerProcess + remainder) * sizeof(Node));

          int added = 0;
          for (int k = 0; k < nodesPerProcess; k++)
          {
            insertNodes[k] = receivedNodes[k * commSize + MPIRank];
            added++;
          }
          // Remainder per each process (if any)
          if (remainder > 0 && MPIRank < remainder)
          {
            insertNodes[nodesPerProcess] = receivedNodes[nodesPerProcess * commSize + MPIRank];
            added++;
          }

          // DEBUGGING
          // if (counter % 100 == 0)
          //   printf("Proc[%d] added = %d at counter[%d]\n", MPIRank, added, counter);

          pushBackBulk(&multiPool[0], insertNodes, added);

          // Free allocated memory
          free(sharedNodes);
          free(receivedNodes);
        }
      }

      // Work Stealing by Last Thread
      if (gpuID == D && w == 2)
      {
        // Termination Detection of GPU-accelerated step
        termination_flag = 0;

        // Check multiPool sizes
        for (int i = 0; i < D; i++)
        {
          if (multiPool[i].size > m || poolSizes_all[i] > 0)
            termination_flag = 1;
        }

        MPI_Allgather(&termination_flag, 1, MPI_INT, global_flags, 1, MPI_INT, MPI_COMM_WORLD);

        int termination = 0;
        for (int i = 0; i < commSize; i++)
          termination += global_flags[i];

        // Warning: poolSizes_all has to be rechecked?
        // int sumPoolSizes = 0;
        // for (int i = 0; i < D; i++)
        //   sumPoolSizes += poolSizes_all[i];
        if (termination == 0) // || sumPoolSizes == 0)
          global_termination_flag = 1;

        // If no Termination, we proceed to work sharing/work stealing
        if (!global_termination_flag)
        {
          // Step 1: Check if any process needs work (below threshold)
          // int threshold = commSize * 2 * m * D;
          int needs_work = local_need;
          // for (int i = 0; i < D; i++)
          //{
          //   if (multiPool[i].size > threshold)
          //     needs_work = 0;
          // }
          int all_needs_work[commSize];

          // Exchange information about work need
          MPI_Allgather(&needs_work, 1, MPI_INT, all_needs_work, 1, MPI_INT, MPI_COMM_WORLD);

          // Count how many processes need work
          int needy_count = 0;
          for (int i = 0; i < commSize; i++)
          {
            if (all_needs_work[i])
              needy_count++;
          }

          // Step 2: Determine how much to give to steal request (if this process has work)
          Node *sharedNodes = NULL;
          int sharedSize = 0;
          int halfSizes;
          // Only proceed if some (but not all) processes need work
          if (needy_count > 0 && needy_count < commSize)
          {
            if (!needs_work)
            {
              int victims[D];
              permute(victims, D);
              for (int j = 0; j < D; j++)
              {
                Node *sharedNodesPartial;
                sharedNodesPartial = popBackBulkHalf(&multiPool[victims[j]], m, M, &halfSizes);

                // If halfSizes > 0, there is local data to share
                if (halfSizes > 0)
                {
                  if (sharedNodes == NULL)
                    sharedNodes = (Node *)malloc(halfSizes * sizeof(Node));
                  else
                    sharedNodes = (Node *)realloc(sharedNodes, (sharedSize + halfSizes) * sizeof(Node));

                  // Optimization (?)
                  // memcpy(sharedNodes, parents + (poolSize - halfSize), halfSize * sizeof(Node));
                  for (int k = 0; k < halfSizes; k++)
                    sharedNodes[sharedSize + k] = sharedNodesPartial[k];
                  sharedSize += halfSizes;
                  free(sharedNodesPartial);
                  break;
                }
              }
            }
          }

          // Step 3: Gather the sizes of the shared data from all processes
          int sendCounts[commSize];
          int recvCounts[commSize];
          int recvDispls[commSize];

          // Each process sends its sharedSize to all other processes
          MPI_Allgather(&sharedSize, 1, MPI_INT, sendCounts, 1, MPI_INT, MPI_COMM_WORLD);

          // Step 4: Compute displacements for the received data
          int totalReceived = 0;
          for (int i = 0; i < commSize; i++)
          {
            recvCounts[i] = sendCounts[i];
            recvDispls[i] = totalReceived;
            totalReceived += recvCounts[i];
          }

          // Step 5: Allocate a buffer to store all received data
          Node *receivedNodes = (Node *)malloc(totalReceived * sizeof(Node));
          if (receivedNodes == NULL)
          {
            fprintf(stderr, "Proc[%d] Thread[%d] Memory allocation failed for receivedNodes\n", MPIRank, gpuID);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
          }

          // Step 6: Gather all data into the receivedNodes buffer
          MPI_Allgatherv(sharedNodes, sharedSize, myNode,
                         receivedNodes, recvCounts, recvDispls, myNode,
                         MPI_COMM_WORLD);

          // Step 6: Redistribute nodes (only for processes that needed work)
          if (needs_work && totalReceived > 0)
          {
            nStealsProc[MPIRank]++;
            int nodesPerProcess = totalReceived / needy_count; // Number of nodes each process will recover from every other process
            int remainder = totalReceived % needy_count;       // Remainder to handle uneven distribution

            // Nodes process per each process
            Node *insertNodes = (Node *)malloc((nodesPerProcess + remainder) * sizeof(Node));

            int added = 0;
            // Calculate position in the list of needy processes
            int needy_position = 0;
            for (int i = 0; i < MPIRank; i++)
            {
              if (all_needs_work[i])
                needy_position++;
            }
            // Nodes recovered by this needy process
            for (int k = 0; k < nodesPerProcess; k++)
            {
              insertNodes[k] = receivedNodes[k * needy_count + needy_position];
              added++;
            }
            // Remainder of nodes recovered per each process (if any)
            if (remainder > 0 && needy_position < remainder)
            {
              insertNodes[nodesPerProcess] = receivedNodes[nodesPerProcess * needy_count + needy_position];
              added++;
            }

            // TODO: improve insertion of stolen nodes into multiple local pools
            // // Total amount of nodes received locally is 'added'
            // int nbPool; // Amount of local pools receiving nodes from WS
            // for (nbPool = D; nbPool >= 1; nbPool--)
            // {
            //   if (added >= nbPool * m)
            //     break;
            // }

            // int nodesPerPool = added / nbPool;
            // int remainderPool = added % nbPool;

            // for (int k = 0; k < nbPool - 1; k++)
            // {
            //   pushBackBulkFree(&multiPool[k], insertNodes + (nodesPerPool * k), nodesPerPool);
            //   atomic_store(&eachTaskState[k], BUSY);
            // }
            // pushBackBulkFree(&multiPool[nbPool - 1], insertNodes + (nodesPerPool * (nbPool - 1)), nodesPerPool + remainderPool);
            // atomic_store(&eachTaskState[nbPool - 1], BUSY);

            pushBackBulk(&multiPool[0], insertNodes, added);
            atomic_store(&eachTaskState[0], BUSY);
            local_need = 0;
            free(insertNodes);
          }

          // Free allocated memory
          free(sharedNodes);
          free(receivedNodes);
        }
      }

      if (gpuID != D)
      {
        /*
          Each task gets its parenst nodes from the pool
        */
        // counter++;
        int poolSize = popBackBulk(pool_loc, m, M, parents);
        poolSizes_all[gpuID] = poolSize;

        if (poolSize > 0)
        {
          if (taskState == IDLE)
          {
            taskState = BUSY;
            atomic_store(&eachTaskState[gpuID], BUSY);
          }

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
          // numBounds is the 'size' of the problem
          startKernelCall = omp_get_wtime();
          evaluate_gpu(jobs, lb, numBounds, nbBlocks, poolSize, best, lbound1_d, lbound2_d, parents_d, bounds_d, sumOffSets_d, nodeIndex_d);
          cudaDeviceSynchronize();
          endKernelCall = omp_get_wtime();
          timeLocalKernelCall[gpuID] += endKernelCall - startKernelCall;
          cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);

          /*
            Each task generates and inserts its children nodes to the pool.
          */
          generate_children(parents, children, poolSize, jobs, bounds, &tree, &sol, &best_l, pool_loc, &indexChildren);
          pushBackBulk(pool_loc, children, indexChildren);
        }
        else
        {
          startTimeIdle = omp_get_wtime();
          // local work stealing
          int tries = 0;
          bool steal = false;
          int victims[D];
          permute(victims, D);

          while (tries < D && steal == false)
          { // WS0 loop
            const int victimID = victims[tries];

            if (victimID != gpuID)
            { // if not me
              SinglePool_atom *victim;
              victim = &multiPool[victimID];
              nSteal++;
              int nn = 0;
              // int count = 0;
              while (nn < 10)
              { // WS1 loop
                expected = false;
                // count++;
                if (atomic_compare_exchange_strong(&(victim->lock), &expected, true))
                { // get the lock
                  int size = victim->size;
                  int nodeSize = 0;

                  if (size >= 2 * m)
                  {
                    Node *p = popBackBulkFree(victim, m, M, &nodeSize);

                    if (nodeSize == 0)
                    {                                       // safety check
                      atomic_store(&(victim->lock), false); // reset lock
                      printf("\nDEADCODE\n");
                      exit(-1);
                    }

                    /* for i in 0..#(size/2) {
                      pool_loc.pushBack(p[i]);
                    } */

                    pushBackBulk(pool_loc, p, nodeSize);

                    steal = true;
                    nSSteal++;
                    atomic_store(&(victim->lock), false); // reset lock
                    endTimeIdle = omp_get_wtime();
                    timeIdleDevice[gpuID] += endTimeIdle - startTimeIdle;
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
          endTimeIdle = omp_get_wtime();
          timeIdleDevice[gpuID] += endTimeIdle - startTimeIdle;
        WS0:
          if (steal == false)
          {
            // termination
            if (taskState == BUSY)
            {
              taskState = IDLE;
              atomic_store(&eachTaskState[gpuID], IDLE);
            }
            if (allIdle(eachTaskState, D, &allTasksIdleFlag))
            {
              // Set request for work while checking global termination
              // request++;
              // while (request < D)
              //{
              //  continue;
              //}
              if (w == 2)
              {
                local_need = 1;
                // atomic_store(&(pool_loc->lock), false);
                while (!global_termination_flag && local_need)
                {
                  if (global_termination_flag == 1)
                    break;
                  if (local_need == 0)
                  {
                    // request = 0;
                    break;
                  }
                }
                continue;
              }
              else if (w == 1)
                continue;
              else
                break;
            }
            continue;
          }
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
      if (gpuID != D)
      {
        const int poolLocSize = pool_loc->size;
        for (int i = 0; i < poolLocSize; i++)
        {
          int hasWork = 0;

          pushBack(&pool_lloc, popBack(pool_loc, &hasWork));
          if (!hasWork)
            break;
        }
      }
    }
    if (gpuID != D)
    {
      eachExploredTree[gpuID] = tree;
      eachExploredSol[gpuID] = sol;
      eachBest[gpuID] = best_l;

      deleteSinglePool_atom(pool_loc);
    }

  } // End of parallel region OpenMP

  MPI_Barrier(MPI_COMM_WORLD);

  /*******************************
  Gathering statistics
  *******************************/

  endTime = omp_get_wtime();
  double t2, t2Temp = endTime - startTime;
  double maxDevice = get_max(timeDevice, D);
  double maxKernelCall = get_max(timeLocalKernelCall, D);
  double maxIdleTime = get_max(timeIdleDevice, D);
  unsigned long long int mySteals = nStealsProc[MPIRank];
  t2Temp -= maxDevice;
  MPI_Reduce(&t2Temp, &t2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // GPU
  for (int i = 0; i < D; i++)
  {
    eachLocaleExploredTree += eachExploredTree[i];
    eachLocaleExploredSol += eachExploredSol[i];
  }
  eachLocaleBest = findMin(eachBest, D);

  // MPI
  unsigned long long int midExploredTree = 0, midExploredSol = 0;
  MPI_Reduce(&eachLocaleExploredTree, &midExploredTree, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&eachLocaleExploredSol, &midExploredSol, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  // TODO : fix this (it should be an All_reduce I suppose)
  MPI_Reduce(&eachLocaleBest, best, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  // Gather data from all processes for printing GPU workload statistics
  unsigned long long int *allExploredTrees = NULL;
  unsigned long long int *allExploredSols = NULL;
  unsigned long long int *allEachExploredTrees = NULL; // For eachExploredTree array
  unsigned long long int *allStealsProc = NULL;        // For eachExploredTree array
  double *allMaxKernelCall = NULL;
  double *allMaxIdleDevice = NULL;
  if (MPIRank == 0)
  {
    *exploredTree += midExploredTree;
    *exploredSol += midExploredSol;
    allExploredTrees = (unsigned long long int *)malloc(commSize * sizeof(unsigned long long int));
    allExploredSols = (unsigned long long int *)malloc(commSize * sizeof(unsigned long long int));
    allEachExploredTrees = (unsigned long long int *)malloc(commSize * D * sizeof(unsigned long long int));
    allStealsProc = (unsigned long long int *)malloc(commSize * sizeof(unsigned long long int));
    allMaxKernelCall = (double *)malloc(commSize * sizeof(double));
    allMaxIdleDevice = (double *)malloc(commSize * sizeof(double));
  }

  MPI_Gather(&eachLocaleExploredTree, 1, MPI_UNSIGNED_LONG_LONG, allExploredTrees, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(&eachLocaleExploredSol, 1, MPI_UNSIGNED_LONG_LONG, allExploredSols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  // Gather eachExploredTree array from all processes
  MPI_Gather(eachExploredTree, D, MPI_UNSIGNED_LONG_LONG, allEachExploredTrees, D, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  MPI_Gather(&maxKernelCall, 1, MPI_DOUBLE, allMaxKernelCall, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&maxIdleTime, 1, MPI_DOUBLE, allMaxIdleDevice, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&mySteals, 1, MPI_UNSIGNED_LONG_LONG, allStealsProc, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // Update GPU
  if (MPIRank == 0)
  {
    for (int i = 0; i < commSize; i++)
    {
      expSolProc[i] = allExploredSols[i];
      expTreeProc[i] = allExploredTrees[i];
      workloadProc[i] = (double)100 * allExploredTrees[i] / ((double)*exploredTree);
      timeKernelCall[i] = allMaxKernelCall[i];
      timeIdle[i] = allMaxIdleDevice[i];
      nStealsProc[i] = allStealsProc[i];
    }

    printf("\nSearch on GPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("  Per MPI process: ");
    for (int i = 0; i < commSize; i++)
      printf("[%d] = %llu ", i, allExploredTrees[i]);
    printf("\n");
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("  Per MPI process: ");
    for (int i = 0; i < commSize; i++)
      printf("[%d] = %llu ", i, allExploredSols[i]);
    printf("\n");
    printf("Max Time Kernel Call per MPI process: ");
    for (int i = 0; i < commSize; i++)
      printf("[%d] = %f ", i, timeKernelCall[i]);
    printf("\n");
    printf("Best solution found: %d\n", *best);
    printf("Elapsed time: %f [s]\n\n", t2);
    printf("Workload per GPU per MPI process: \n");
    for (int i = 0; i < commSize; i++)
    {
      printf("  Process [%d]: ", i);
      for (int gpuID = 0; gpuID < D; gpuID++)
        printf("%.2f ", (double)100 * allEachExploredTrees[i * D + gpuID] / ((double)*exploredTree));
      printf("\n");
    }
  }

  /*
    Step 3: Remaining nodes evaluated in DFS on CPU by each MPI process.
  */
  unsigned long long int finalLocaleExpTree = 0, finalLocaleExpSol = 0;
  startTime = omp_get_wtime();
  while (1)
  {
    int hasWork = 0;
    Node parent = popBack(&pool_lloc, &hasWork);
    if (!hasWork)
      break;
    decompose(jobs, lb, best, lbound1, lbound2, parent, &finalLocaleExpTree, &finalLocaleExpSol, &pool_lloc);
  }
  endTime = omp_get_wtime();
  double t3, t3Temp = endTime - startTime;

  MPI_Reduce(&t3Temp, &t3, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  unsigned long long int masterFinalLocaleExpTree = 0, masterFinalLocaleExpSol = 0;
  MPI_Reduce(&finalLocaleExpTree, &masterFinalLocaleExpTree, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&finalLocaleExpSol, &masterFinalLocaleExpSol, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  // TODO : Fix this one also
  // MPI_Reduce(&eachLocaleBest, best, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  *elapsedTime = t1 + t2 + t3;

  // freeing memory for structs common to all MPI processes
  deleteSinglePool_atom(&pool);
  deleteSinglePool_atom(&pool_lloc);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);

  if (MPIRank == 0)
  {
    *exploredTree += masterFinalLocaleExpTree;
    *exploredSol += masterFinalLocaleExpSol;

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
  // Distributed Multi-GPU PFSP only uses: inst, lb, ub, m, M, D, ws, LB
  int inst, lb, ub, m, M, D, ws, LB;
  double perc;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &D, &ws, &LB, &perc);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  if (MPIRank == 0)
    print_settings(inst, machines, jobs, ub, lb, D, ws, commSize, LB, version);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;

  unsigned long long int exploredTree = 0, exploredSol = 0, expTreeProc[commSize], expSolProc[commSize], nStealsProc[commSize];
  double elapsedTime, timeKernelCall[commSize], timeIdle[commSize], workloadProc[commSize];

  for (int i = 0; i < commSize; i++)
  {
    expTreeProc[i] = 0;
    expSolProc[i] = 0;
    timeKernelCall[i] = 0;
    timeIdle[i] = 0;
    workloadProc[i] = 0;
    nStealsProc[i] = 0;
  }

  pfsp_search(inst, lb, m, M, D, LB, perc, &optimum, &exploredTree, &exploredSol, &elapsedTime, expTreeProc, expSolProc, nStealsProc, timeKernelCall, timeIdle, workloadProc, MPIRank, commSize);

  if (MPIRank == 0)
  {
    print_results(optimum, exploredTree, exploredSol, elapsedTime);
    print_results_file(inst, machines, jobs, lb, D, LB, commSize, optimum, exploredTree, exploredSol, elapsedTime, expTreeProc, expSolProc, nStealsProc, timeKernelCall, timeIdle, workloadProc);
  }

  MPI_Finalize();

  return 0;
}