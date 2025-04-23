/*
  Distributed Single-GPU B&B to solve Taillard instances of the PFSP in C+MPI+CUDA.
*/

// for k in {29,30,22,27,23}; do for i in {1..3}; do mpirun --hostfile hostfile --map-by ppr:1:node -mca pml ucx --mca btl ^openib -np 1 ./pfsp_dist_gpu_cuda.out -i ${k} -l 2 -m 25; done; done;

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
#include <mpi.h>
// #include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"
#include "lib/evaluate.h"
#include "lib/Pool_ext.h"
#include "lib/Auxiliary.h"

/******************************************************************************
Create new MPI data type
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
CUDA error checking
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

void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb, const int commSize)
{
  printf("\n=================================================\n");
  printf("Distributed Single-GPU C+MPI+CUDA (%d MPI processes)\n\n", commSize);
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

void print_results_file(const int inst, const int machines, const int jobs, const int lb, const int commSize, const int optimum,
                        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer)
{
  FILE *file;
  file = fopen("stats_pfsp_dist_gpu_cuda.dat", "a");
  fprintf(file, "ta%d lb%d [%d]Proc S-GPU %.4f %llu %llu %d\n", inst, lb, commSize, timer, exploredTree, exploredSol, optimum);
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
                   int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_ext *pool)
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
                     int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_ext *pool)
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
                   SinglePool_ext *pool)
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
               unsigned long long int *num_sol, SinglePool_ext *pool)
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
                       unsigned long long int *exploredTree, unsigned long long int *exploredSol, int *best, SinglePool_ext *pool)
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
void pfsp_search(const int inst, const int lb, const int m, const int M, int *best,
                 unsigned long long int *exploredTree, unsigned long long int *exploredSol,
                 double *elapsedTime, int MPIRank, int commSize)
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

  SinglePool_ext pool;
  initSinglePool_ext(&pool);

  pushBack(&pool, root);

  // Timer
  // double startTime, endTime;
  // startTime = omp_get_wtime();
  // Timer
  struct timespec startTime, endTime; // startMeasure, endMeasure;
  clock_gettime(CLOCK_MONOTONIC_RAW, &startTime);

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

  // endTime = omp_get_wtime();
  clock_gettime(CLOCK_MONOTONIC_RAW, &endTime);
  double t1Temp = (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec) / 1e9;
  double t1;
  // double t1, t1Temp = endTime - startTime;
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

  // startTime = omp_get_wtime();
  clock_gettime(CLOCK_MONOTONIC_RAW, &startTime);

  const int poolSize = pool.size;
  const int c = poolSize / commSize;
  const int l = poolSize - (commSize - 1) * c;
  const int f = pool.front;

  pool.front = 0;
  pool.size = 0;

  // For every MPI process
  unsigned long long int eachLocaleExploredTree = 0, eachLocaleExploredSol = 0;
  int eachLocaleBest = *best;

  // One per compute node (needs to be communicated between processes)
  _Atomic bool localeState = false;
  // WARNING: For termination detection of distributed step
  _Atomic bool allLocalesIdleFlag = false;

  SinglePool_ext pool_lloc;
  initSinglePool_ext(&pool_lloc);

  // Variables for One-Sided Work-Stealing using only MPI_Put

  // TO DO: Window Info
  // MPI_Info info_nodes, info_requests;

  int *steal_request;
  MPI_Win win_requests;
  int err = MPI_Win_allocate(commSize * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &steal_request, &win_requests);

  if (err != MPI_SUCCESS || steal_request == NULL)
  {
    char err_string[MPI_MAX_ERROR_STRING];
    int err_length;
    MPI_Error_string(err, err_string, &err_length);
    fprintf(stderr, "MPI_Win_allocate for requests failed: %s\n", err_string);
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  int *steal_responses;
  MPI_Win win_responses;
  err = MPI_Win_allocate(commSize * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &steal_responses, &win_responses);

  if (err != MPI_SUCCESS || steal_responses == NULL)
  {
    char err_string[MPI_MAX_ERROR_STRING];
    int err_length;
    MPI_Error_string(err, err_string, &err_length);
    fprintf(stderr, "MPI_Win_allocate for requests failed: %s\n", err_string);
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  int *steal_sizes;
  MPI_Win win_sizes;
  err = MPI_Win_allocate(commSize * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &steal_sizes, &win_sizes);

  if (err != MPI_SUCCESS || steal_sizes == NULL)
  {
    char err_string[MPI_MAX_ERROR_STRING];
    int err_length;
    MPI_Error_string(err, err_string, &err_length);
    fprintf(stderr, "MPI_Win_allocate for sizes failed: %s\n", err_string);
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  int capacity_per_proc = 300000;
  Node *steal_nodes;
  MPI_Win win_nodes;
  err = MPI_Win_allocate(commSize * capacity_per_proc * sizeof(Node), sizeof(Node), MPI_INFO_NULL, MPI_COMM_WORLD, &steal_nodes, &win_nodes);

  if (err != MPI_SUCCESS || steal_nodes == NULL)
  {
    char err_string[MPI_MAX_ERROR_STRING];
    int err_length;
    MPI_Error_string(err, err_string, &err_length);
    fprintf(stderr, "MPI_Win_allocate for nodes failed: %s\n", err_string);
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  for (int i = 0; i < commSize; i++)
  {
    // From 0 to 'commSize - 1' indicates which MPIRank contributed with work
    // '-2' indicates the need of work
    steal_request[i] = 0; // BUSY
    steal_responses[i] = 0;
    steal_sizes[i] = 0;
  }

  // MPI_Win_sync(win_sizes);
  // MPI_Win_sync(win_requests);
  // MPI_Win_sync(win_nodes);
  MPI_Barrier(MPI_COMM_WORLD);

  // each MPI process gets its chunk
  for (int i = 0; i < c; i++)
  {
    pool_lloc.elements[i] = pool.elements[MPIRank + f + i * commSize];
  }
  pool_lloc.size += c;
  if (MPIRank == commSize - 1)
  {
    for (int i = c; i < l; i++)
    {
      pool_lloc.elements[i] = pool.elements[(commSize * c) + f + i - c];
    }
    pool_lloc.size += l - c;
  }

  // EVENTUAL USEFUL VARIABLES
  // cudaSetDevice(0);
  // int nSteal = 0, nSSteal = 0;
  // int best_l = *best;
  // bool expected = false;

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

  int nSteal = 0;
  int nSSteal = 0;
  int stolen = 0;

  int amount_nodes = 0;
  int counter = 0;
  // double time1, time2, time1Idle, time2Idle, put_sizes = 0, put_request = 0, put_nodes = 0, compare_request = 0, sync_sizes = 0,
  //   time1GPUWork, time2GPUWork, GPUWork = 0, GPUTime = 0, IdleTime = 0, cpyParents = 0, cpyBounds = 0, cpyKernel = 0;

  // cudaSetDevice(MPIRank);

  printf("Proc[%d] I'm all good just before the while\n", MPIRank);
  while (1)
  {
    counter++;
    int poolSize = popBackBulk(&pool_lloc, m, M, parents);
    if (poolSize > 0)
    {
      // time1GPUWork = omp_get_wtime();

      if (atomic_load(&localeState) == IDLE)
        atomic_store(&localeState, BUSY);

      /*
        TODO: Optimize 'numBounds' based on the fact that the maximum number of
        generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
        something like that.
      */
      const int numBounds = jobs * poolSize;
      const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);

      cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);
      evaluate_gpu(jobs, lb, numBounds, nbBlocks, &eachLocaleBest, lbound1_d, lbound2_d, parents_d, bounds_d);
      cudaDeviceSynchronize();
      cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);

      /*
        each task generates and inserts its children nodes to the pool.
      */
      generate_children(parents, poolSize, jobs, bounds,
                        &eachLocaleExploredTree, &eachLocaleExploredSol, best, &pool_lloc);

      // Answer WS requests after a complete round of bounding+pruning+branching
      int flagShared = 0;
      int flagNonShared = -1;
      int requests[commSize];
      permute(requests, commSize); // Introduce some randomness

      MPI_Win_sync(win_requests);
      MPI_Win_sync(win_responses);
      for (int tries = 0; tries < commSize; tries++)
      {
        int requestID = requests[tries];

        if (requestID == MPIRank)
          continue;

        // Validate request ID
        if (requestID < 0 || requestID >= commSize)
        {
          printf("Rank %d: Invalid requestID %d\n", MPIRank, requestID);
          continue;
        }

        if (steal_request[requestID] == 1)
        {
          if (pool_lloc.size < 2 * m || flagShared == 1) // Reject request
          {
            nSteal++;

            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, requestID, 0, win_responses);
            MPI_Put(&flagNonShared, 1, MPI_INT, requestID, MPIRank, 1, MPI_INT, win_responses);
            MPI_Win_flush(requestID, win_responses);
            // MPI_Win_flush_all(win_responses);
            MPI_Win_unlock(requestID, win_responses);
          }
          else if (flagShared == 0) // Accept request
          {
            flagShared = 1;
            // ATTENTION : If work stealing is succesful this should come back
            // steal_request[requestID] = 0;
            (pool_lloc.size > 2 * capacity_per_proc) ? (amount_nodes = capacity_per_proc) : (amount_nodes = pool_lloc.size / 2);
            stolen += amount_nodes;
            printf("Proc[%d] Gives amount_nodes = %d to RequestProc[%d], counter=%d\n", MPIRank, amount_nodes, requestID, counter);

            // Node *nodes = popBackBulkFreeN(&pool_lloc, m, M, &amount_nodes);
            // if (amount_nodes == 0)
            // {
            //   printf("ERROR Pool_lloc size is 0 on PopBackBulkFreeN\n");
            //   exit(-1);
            // }

            // // time1 = omp_get_wtime();
            // MPI_Win_lock(MPI_LOCK_EXCLUSIVE, requestID, 0, win_nodes);
            // MPI_Put(nodes, amount_nodes * sizeof(Node), MPI_BYTE, requestID, MPIRank * capacity_per_proc, amount_nodes * sizeof(Node), MPI_BYTE, win_nodes);
            // MPI_Win_flush(requestID, win_nodes);
            // // MPI_Win_flush_all(win_nodes);
            // MPI_Win_unlock(requestID, win_nodes);

            // // time2 = omp_get_wtime();
            // // put_nodes += time2 - time1;
            // // time1 = omp_get_wtime();

            // MPI_Win_lock(MPI_LOCK_EXCLUSIVE, requestID, 0, win_sizes);
            // MPI_Put(&amount_nodes, 1, MPI_INT, requestID, MPIRank, 1, MPI_INT, win_sizes);
            // // MPI_Win_flush(requestID, win_sizes);
            // MPI_Win_flush_all(win_sizes);
            // MPI_Win_unlock(requestID, win_sizes);

            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, requestID, 0, win_responses);
            MPI_Put(&flagShared, 1, MPI_INT, requestID, MPIRank, 1, MPI_INT, win_responses);
            MPI_Win_flush(requestID, win_responses);
            // MPI_Win_flush_all(win_responses);
            MPI_Win_unlock(requestID, win_responses);

            // time2 = omp_get_wtime();
            // put_sizes += time2 - time1;

            printf("After Put/Send: Proc[%d] gave [%d] work to RequestProc[%d] counter=%d\n", MPIRank, amount_nodes, requestID, counter);
            nSSteal++;
          }
        }
      }
      // time2GPUWork = omp_get_wtime();
      // GPUWork += time2GPUWork - time1GPUWork;
    }
    else
    {
      // time1Idle = omp_get_wtime();
      //  One MPI process = no Work-Stealing
      if (commSize == 1)
        break;

      // Distributed Work-Stealing per request
      bool remoteSteal = true;
      steal_request[MPIRank] = 1; // '1' indicates steal request
      printf("Proc [%d] Sends steal request, counter=%d\n", MPIRank, counter);

      // time1 = omp_get_wtime();
      for (int j = 0; j < commSize; j++) // Asking for work
      {
        if (j != MPIRank)
        {
          MPI_Win_lock(MPI_LOCK_EXCLUSIVE, j, 0, win_requests);
          MPI_Put(&steal_request[MPIRank], 1, MPI_INT, j, MPIRank, 1, MPI_INT, win_requests);
          // MPI_Win_flush(j, win_requests);
          MPI_Win_flush_all(win_requests);
          MPI_Win_unlock(j, win_requests);
        }
      }
      // time2 = omp_get_wtime();
      // put_request += time2 - time1;

      while (steal_request[MPIRank] == 1) // I have no work
      {
        MPI_Win_sync(win_requests);

        int sum = 0;
        for (int checkRequests = 0; checkRequests < commSize; checkRequests++)
        {
          sum += steal_request[checkRequests]; // Check if none has work anymore
          // if (checkRequests != MPIRank)        // I skip myself
          //   continue;
          // if (steal_request[checkRequests] == 1) // I tell others asking work I cannot share work
          // {
          //   int flagNonShared = -1;
          //   MPI_Win_lock(MPI_LOCK_EXCLUSIVE, checkRequests, 0, win_responses);
          //   MPI_Put(&flagNonShared, 1, MPI_INT, checkRequests, MPIRank, 1, MPI_INT, win_responses);
          //   // MPI_Win_flush(checkRequests, win_responses);
          //   MPI_Win_flush_all(win_responses);
          //   MPI_Win_unlock(checkRequests, win_responses);
          //   steal_request[checkRequests] = 0; // I ignore who asks me work
          // }
        }

        int countResponses = 0;
        for (int checkReponses = 0; checkReponses < commSize; checkReponses++)
        {
          if (checkReponses == MPIRank)
            continue;
          if (steal_responses[checkReponses] == -1 || steal_responses[checkReponses] == 1)
            countResponses++;
        }

        countResponses = 0; // REMOVE THIS AFTER DEBUGGING
        if (countResponses == commSize - 1)
        {
          // MPI_Recv(&steal_sizes[responder], 1, MPI_INT, responder, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // printf("Proc[%d] got [%d] work from StolenProc[%d] counter=%d\n", MPIRank, steal_sizes[responder], responder, counter);
          for (int i = 0; i < commSize; i++)
          {
            if (i == MPIRank || steal_responses[i] == -1)
            {
              steal_responses[i] = 0;
              continue;
            }
            int responseSize = steal_sizes[i];
            // pushBackBulk(&pool_lloc, (steal_nodes + (i * capacity_per_proc)), responseSize);
            steal_responses[i] = 0;
          }
          // steal_sizes[responder] = 0;
          steal_request[MPIRank] = 0;

          // printf("Proc[%d] steal_request[%d] = %d after responder counter=%d\n", MPIRank, i, steal_request[i], counter);

          // This line below should be back
          // MPI_Recv((steal_nodes + (responder * capacity_per_proc)), steal_sizes[responder] * sizeof(Node), MPI_BYTE, responder, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          // time1 = omp_get_wtime();

          // while (steal_sizes[responder] == 0)
          // {
          //   //MPI_Win_sync(win_requests);
          //   //MPI_Win_sync(win_nodes);

          //   counter++;
          //   if (counter % 500 == 0)
          //     MPI_Win_sync(win_sizes);

          //   if (counter % 1000000 == 0)
          //   {
          //     for (int q = 0; q < commSize; q++)
          //       printf("With counter: Proc[%d] steal_sizes[%d] = %d\n", MPIRank, q, steal_sizes[q]);
          //     printf("ERROR: I Proc[%d] never receive work\n", MPIRank);
          //     // exit(-1);
          //   }
          //   // continue;
          // }
          // time2 = omp_get_wtime();

          // sync_sizes += time2 - time1;

          // printf("After While: Proc[%d] got [%d] work from StolenProc[%d], counter = %d\n", MPIRank, steal_sizes[responder], responder, counter);
          //  pushBackBulk(&pool_lloc, (steal_nodes + (responder * capacity_per_proc)), steal_sizes[responder]);
        }
        if (steal_request[MPIRank] == 0)
          break;
        if (sum == commSize) // Everybody has no more work, then we should be over
        {
          remoteSteal = false;
          break;
        }
      }

      // Distributed Termination Condition
      if (remoteSteal == false)
      {
        printf("Proc[%d] Are we all done ?\n", MPIRank);
        MPI_Barrier(MPI_COMM_WORLD);
        //  MPI_Win_fence(0, win_requests);
        //  MPI_Win_fence(0, win_requests);
        break;

        // Old termination condition
        // This termination condition is not 'ideal' for distributed level
        // if (atomic_load(&localeState) == BUSY)
        //   atomic_store(&localeState, IDLE);

        // bool *allLocaleStateTemp = (bool *)malloc(commSize * sizeof(bool));
        // _Atomic bool *allLocaleState = (_Atomic bool *)malloc(commSize * sizeof(_Atomic bool));
        // bool eachLocaleStateTemp = atomic_load(&localeState);

        // // Gather boolean states from all processes to all processes
        // MPI_Allgather(&eachLocaleStateTemp, 1, MPI_C_BOOL, allLocaleStateTemp, 1, MPI_C_BOOL, MPI_COMM_WORLD);
        // for (int i = 0; i < commSize; i++)
        //   atomic_store(&allLocaleState[i], allLocaleStateTemp[i]);

        // // Check if eachLocalState of every process agrees for termination
        // if (allIdle(allLocaleState, commSize, &allLocalesIdleFlag))
        // {
        //   free(allLocaleStateTemp);
        //   free(allLocaleState);
        //   // time2Idle = omp_get_wtime();
        //   // IdleTime += time2Idle - time1Idle;
        //   break;
        // }
        // // time2Idle = omp_get_wtime();
        // // IdleTime += time2Idle - time1Idle;
        // continue;
      }
      else
      {
        // time2Idle = omp_get_wtime();
        // IdleTime += time2Idle - time1Idle;
        continue;
      }
    }
  } // End of GPU-accelerated bounding / Work-Stealing

  for (int i = 0; i < commSize; i++)
    printf("Proc[%d] steal_request[%d] = %d\n", MPIRank, i, steal_request[i]);

  printf("Proc[%d] nSSteal = %d nSteal = %d pool_lloc.size = %d Stolen = [%d]\n", MPIRank, nSSteal, nSteal, pool_lloc.size, stolen);
  // printf("Proc[%d] put_sizes = %.4f, put_request = %.4f, put_nodes = %.4f, compare_request = %.4f, sync_sizes = %.4f\n GPUWork = %.4f, cpyBounds = %.4f, cpyKernel = %.4f, cpyParents = %.4f, GPUTime = %.4f, IdleTime = %.4f\n",
  //  MPIRank, put_sizes, put_request, put_nodes, compare_request, sync_sizes, GPUWork, cpyBounds, cpyKernel, cpyParents, GPUTime, IdleTime);

  MPI_Barrier(MPI_COMM_WORLD);

  // Freeing device and host memory from GPU-accelerated step (step 2)
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

  /*******************************
  Gathering statistics
  *******************************/

  // endTime = omp_get_wtime();
  // double t2, t2Temp = endTime - startTime;
  clock_gettime(CLOCK_MONOTONIC_RAW, &endTime);
  double t2Temp = (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec) / 1e9;
  double t2;

  MPI_Reduce(&t2Temp, &t2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // // GPU
  // for (int i = 0; i < D; i++)
  // {
  //   eachLocaleExploredTree += eachExploredTree[i];
  //   eachLocaleExploredSol += eachExploredSol[i];
  // }
  // eachLocaleBest = findMin(eachBest, D);

  // MPI
  unsigned long long int midExploredTree = 0, midExploredSol = 0;
  MPI_Reduce(&eachLocaleExploredTree, &midExploredTree, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&eachLocaleExploredSol, &midExploredSol, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&eachLocaleBest, best, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  if (MPIRank == 0)
  {
    *exploredTree += midExploredTree;
    *exploredSol += midExploredSol;
  }

  // Gather data from all processes for printing GPU workload statistics
  unsigned long long int *allExploredTrees = NULL;
  unsigned long long int *allExploredSols = NULL;
  // unsigned long long int *allEachExploredTrees = NULL; // For eachExploredTree array
  if (MPIRank == 0)
  {
    allExploredTrees = (unsigned long long int *)malloc(commSize * sizeof(unsigned long long int));
    allExploredSols = (unsigned long long int *)malloc(commSize * sizeof(unsigned long long int));
    // allEachExploredTrees = (unsigned long long int *)malloc(commSize * D * sizeof(unsigned long long int));
  }

  MPI_Gather(&eachLocaleExploredTree, 1, MPI_UNSIGNED_LONG_LONG, allExploredTrees, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(&eachLocaleExploredSol, 1, MPI_UNSIGNED_LONG_LONG, allExploredSols, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // Gather eachExploredTree array from all processes
  // MPI_Gather(eachExploredTree, D, MPI_UNSIGNED_LONG_LONG, allEachExploredTrees, D, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // Update GPU
  if (MPIRank == 0)
  {
    printf("\nSearch on GPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("  Per MPI process: ");
    for (int i = 0; i < commSize; i++)
      printf("%d = %llu ", i, allExploredTrees[i]);
    printf("\n");
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("  Per MPI process: ");
    for (int i = 0; i < commSize; i++)
      printf("%d = %llu ", i, allExploredSols[i]);
    printf("\n");
    printf("Best solution found: %d\n", *best);
    printf("Elapsed time: %f [s]\n\n", t2);
    // printf("Workload per GPU per MPI process: \n");
    // for (int i = 0; i < commSize; i++)
    // {
    //   printf("  Process %d: ", i);
    //   for (int gpuID = 0; gpuID < D; gpuID++)
    //     printf("%.2f ", (double)100 * allEachExploredTrees[i * D + gpuID] / ((double)*exploredTree));
    //   printf("\n");
    // }
  }

  // Gathering remaining nodes
  int *recvcounts = NULL;
  int *displs = NULL;
  int totalSize = 0;

  if (MPIRank == 0)
  {
    recvcounts = (int *)malloc(commSize * sizeof(int));
    displs = (int *)malloc(commSize * sizeof(int));
  }

  MPI_Gather(&pool_lloc.size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (MPIRank == 0)
  {
    displs[0] = 0;
    for (int i = 0; i < commSize; i++)
    {
      totalSize += recvcounts[i];
      if (i > 0)
      {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
      }
    }
  }

  // Master process receiving data from Gather operation
  Node *masterNodes = NULL;
  if (MPIRank == 0)
  {
    masterNodes = (Node *)malloc(totalSize * sizeof(Node));
  }

  MPI_Gatherv(pool_lloc.elements, pool_lloc.size, myNode,
              masterNodes, recvcounts, displs, myNode, 0, MPI_COMM_WORLD);

  if (MPIRank == 0)
  {
    for (int i = 0; i < totalSize; i++)
      pushBack(&pool, masterNodes[i]);
  }
  /*
    Step 3: We complete the depth-first search on CPU.
  */
  if (MPIRank == 0)
  {
    // int count = 0;
    // startTime = omp_get_wtime();
    clock_gettime(CLOCK_MONOTONIC_RAW, &startTime);
    while (1)
    {
      int hasWork = 0;
      Node parent = popBack(&pool, &hasWork);
      if (!hasWork)
        break;
      decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
      // count++;
    }
  }

  // Freeing memory for structs
  deleteSinglePool_ext(&pool);
  deleteSinglePool_ext(&pool_lloc);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);

  if (MPIRank == 0)
  {
    free(recvcounts);
    free(displs);
    free(masterNodes);

    // endTime = omp_get_wtime();
    // double t3 = endTime - startTime;
    clock_gettime(CLOCK_MONOTONIC_RAW, &endTime);
    double t3 = (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec) / 1e9;

    *elapsedTime = t1 + t2 + t3;
    printf("\nSearch on CPU completed\n");
    printf("Size of the explored tree: %llu\n", *exploredTree);
    printf("Number of explored solutions: %llu\n", *exploredSol);
    printf("Elapsed time: %f [s]\n", t3);

    printf("\nExploration terminated.\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Win_free(&win_nodes);
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
    printf("MPI does not support multiple threads.\n");
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

  int inst, lb, ub, m, M;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  if (MPIRank == 0)
  {
    print_settings(inst, machines, jobs, ub, lb, commSize);
  }

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;

  pfsp_search(inst, lb, m, M, &optimum, &exploredTree, &exploredSol, &elapsedTime, MPIRank, commSize);

  if (MPIRank == 0)
  {
    print_results(optimum, exploredTree, exploredSol, elapsedTime);
    print_results_file(inst, machines, jobs, lb, commSize, optimum, exploredTree, exploredSol, elapsedTime);
  }

  MPI_Finalize();
  return 0;
}
