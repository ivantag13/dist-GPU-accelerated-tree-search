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
*****************************************************************************/

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

/***********************************************************************************
Implementation of the parallel Distributed Multi-GPU C+MPI+OpenMP+CUDA PFSP search.
***********************************************************************************/

void parse_parameters(int argc, char *argv[], int *inst, int *lb, int *ub, int *m, int *M, int *D, double *perc)
{
  *m = 25;
  *M = 50000;
  *inst = 14;
  *lb = 1;
  *ub = 1;
  *D = 1;
  *perc = 0.5;
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
      {"D", required_argument, NULL, 'D'},
      {"perc", required_argument, NULL, 'p'},
      {NULL, 0, NULL, 0} // Terminate options array
  };

  int opt, value;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "i:l:u:m:M:D:p:", long_options, &option_index)) != -1)
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

    case 'D':
      if (value < 0)
      {
        fprintf(stderr, "Error: unsupported number of GPU(s)\n");
        exit(EXIT_FAILURE);
      }
      *D = value;
      break;

    case 'p':
      if (value <= 0 || value > 100)
      {
        fprintf(stderr, "Error: unsupported WS percentage for popFrontBulkFree\n");
        exit(EXIT_FAILURE);
      }
      *perc = (double)value / 100;
      break;

    default:
      fprintf(stderr, "Usage: %s --inst <value> --lb <value> --ub <value> --m <value> --M <value> --D <value> --perc <value>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb, const int D, const int numProcs)
{
  printf("\n=================================================\n");
  printf("Distributed multi-GPU C+MPI+OpenMP+CUDA (%d MPI processes x %d GPUs)\n\n", numProcs, D);
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

void print_results_file(const int inst, const int machines, const int jobs, const int lb, const int D, const int commSize, const int optimum,
                        const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer,
                        unsigned long long int *expTreeProc, unsigned long long int *expSolProc, unsigned long long int *nStealsProc,
                        double *timeKernelCall, double *timeIdle, double *workloadProc)
{
  FILE *file;
  file = fopen("distMultigpu_sharing.dat", "a");
  fprintf(file, "\nProc[%d] GPU[%d] ta[%d] lb[%d] time[%.4f] Tree[%llu] Sol[%llu] Best[%d]\n", commSize, D, inst, lb, timer, exploredTree, exploredSol, optimum);
  fprintf(file, "Workload per Proc: ");
  compute_boxplot_stats(workloadProc, commSize, file);
  fprintf(file, "Max Kernel Call Times per Proc: ");
  compute_boxplot_stats(timeKernelCall, commSize, file);
  fprintf(file, "Max Idle Times per Proc: ");
  compute_boxplot_stats(timeIdle, commSize, file);
  fprintf(file, "Work Sharing per Proc: ");
  for (int i = 0; i < commSize; i++)
  {
    if (i != commSize - 1)
      fprintf(file, "%llu ", nStealsProc[i]);
    else
      fprintf(file, "%llu\n", nStealsProc[i]);
  }
  fclose(file);

  file = fopen("distMultigpu_sharing_detail.dat", "a");
  fprintf(file, "\nProc[%d] GPU[%d] ta%d lb%d time[%.4f] Tree[%llu] Sol[%llu] Best[%d]\n", commSize, D, inst, lb, timer, exploredTree, exploredSol, optimum);
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
  fprintf(file, "Succesful Work Sharing per Proc: ");
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

void decompose(const int jobs, const int lb, int *best, const lb1_bound_data *const lbound1, const lb2_bound_data *const lbound2, const Node parent, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_ext *pool)
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
void generate_children(Node *parents, Node *children, const int size, const int jobs, int *bounds, unsigned long long int *exploredTree,
                       unsigned long long int *exploredSol, int *best, SinglePool_ext *pool, int *index)
{
  int sum = 0;
  int childrenIndex = 0;
  for (int i = 0; i < size; i++)
  {
    Node parent = parents[i];
    const uint8_t depth = parent.depth;
    const int limit1 = parent.limit1;

    for (int j = limit1 + 1; j < jobs; j++)
    {
      const int lowerbound = bounds[(j - (limit1 + 1) + sum)];

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
          children[childrenIndex] = child;
          childrenIndex++;

          // pushBack(pool, child);
          *exploredTree += 1;
        }
      }
    }
    sum += jobs - depth;
  }
  *index = childrenIndex;
}

// Distributed Multi-GPU PFSP search
void pfsp_search(const int inst, const int lb, const int m, const int M, const int D, double perc, int *best,
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

  SinglePool_ext pool;
  initSinglePool_ext(&pool);

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

  // TODO: Do step 1 only for master thread?
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

  const int poolSize = pool.size;
  const int c = poolSize / commSize;
  const int l = poolSize - (commSize - 1) * c;
  const int f = pool.front;

  pool.front = 0;
  pool.size = 0;

  // For every MPI process
  unsigned long long int eachLocaleExploredTree = 0, eachLocaleExploredSol = 0;
  int eachLocaleBest = *best;

  // For GPUs under each MPI process
  unsigned long long int eachExploredTree[D], eachExploredSol[D];
  int eachBest[D];

  SinglePool_ext pool_lloc;
  initSinglePool_ext(&pool_lloc);

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

  // Variables for GPUs under each MPI process
  const int poolSize_l = pool_lloc.size;
  const int c_l = poolSize_l / D;
  const int l_l = poolSize_l - (D - 1) * c_l;
  const int f_l = pool_lloc.front;

  pool_lloc.front = 0;
  pool_lloc.size = 0;

  SinglePool_ext multiPool[D];
  for (int i = 0; i < D; i++)
    initSinglePool_ext(&multiPool[i]);

  // Boolean variables for termination detection
  _Atomic bool allTasksIdleFlag = false;
  _Atomic bool eachTaskState[D]; // one task per GPU
  for (int i = 0; i < D; i++)
    atomic_store(&eachTaskState[i], BUSY);

  // TODO: Implement OpenMP reduction to variables best_l, eachExploredTree, eachExploredSol
  // int best_l = *best;
  int global_termination_flag = 0;
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

#pragma omp parallel num_threads(D + 1) shared(eachExploredTree, eachExploredSol, eachBest, eachTaskState, allTasksIdleFlag,  \
                                                   pool_lloc, multiPool, jobs, machines, lbound1, lbound2, lb, m, M, D, perc, \
                                                   best, exploredTree, exploredSol, global_termination_flag, poolSizes_all, timeDevice) // reduction(min:best_l)
  // for (int gpuID = 0; gpuID < D; gpuID++)
  {
    double startSetDevice, endSetDevice, startKernelCall, endKernelCall, startTimeIdle, endTimeIdle;
    int nSteal = 0, nSSteal = 0;
    int gpuID = omp_get_thread_num();

    // DEBUGGING
    // printf("From Proc[%d] Thread[%d] Started MPI+Threading\n", MPIRank, gpuID);

    // WARNING: gpuID == D does not managed a GPU!!!
    if (gpuID != D)
    {
      startSetDevice = omp_get_wtime();
      cudaSetDevice(gpuID);
      endSetDevice = omp_get_wtime();
      double timeSetDevice = endSetDevice - startSetDevice;
      timeDevice[gpuID] = timeSetDevice;
    }

    unsigned long long int tree = 0, sol = 0;
    SinglePool_ext *pool_loc;
    if (gpuID != D)
      pool_loc = &multiPool[gpuID];
    int best_l = *best;
    bool taskState = BUSY;
    bool expected = false;

    if (gpuID != D)
    {
      // each task gets its chunk
      for (int i = 0; i < c_l; i++)
      {
        pool_loc->elements[i] = pool_lloc.elements[gpuID + f_l + i * D];
      }
      pool_loc->size += c_l;
      if (gpuID == D - 1)
      {
        for (int i = c_l; i < l_l; i++)
        {
          pool_loc->elements[i] = pool_lloc.elements[(D * c_l) + f_l + i - c_l];
        }
        pool_loc->size += l_l - c_l;
      }
    }
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
    cudaMemcpy(p_times_d, lbound1->p_times, jobs * machines * sizeof(int), cudaMemcpyHostToDevice);
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
    cudaMalloc((void **)&johnson_schedule_d, nb_mac_pairs * jobs * sizeof(int));
    cudaMalloc((void **)&lags_d, nb_mac_pairs * jobs * sizeof(int));
    cudaMalloc((void **)&machine_pairs_1_d, nb_mac_pairs * sizeof(int));
    cudaMalloc((void **)&machine_pairs_2_d, nb_mac_pairs * sizeof(int));
    cudaMalloc((void **)&machine_pair_order_d, nb_mac_pairs * sizeof(int));
    cudaMemcpy(johnson_schedule_d, lbound2->johnson_schedules, nb_mac_pairs * jobs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(lags_d, lbound2->lags, nb_mac_pairs * jobs * sizeof(int), cudaMemcpyHostToDevice);
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
      if (global_termination_flag)
      {
        // DEBUGGING
        // printf("From Proc[%d] Thread[%d] Global Termination Reached\n", MPIRank, gpuID);
        break;
      }

      // Work Sharing by Last Thread
      if (gpuID == D)
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
          for (int j = 0; j < D; j++)
          {
            Node *sharedNodesPartial;
            sharedNodesPartial = popBackBulkHalf(&multiPool[j], m, M, &halfSizes);

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
            }
          }
          if(sharedSize > 0)
            nStealsProc[MPIRank]++;

          int sendCounts[commSize];
          int recvCounts[commSize];
          int recvDispls[commSize];

          // Step 2: Gather the sizes of the shared data (sharedSize) from all processess
          MPI_Allgather(&sharedSize, 1, MPI_INT, sendCounts, 1, MPI_INT, MPI_COMM_WORLD);

          //DEBUGGING
          // if (counter % 100 == 0)
          //   printf("Proc[%d] sharedSize = %d at counter[%d]\n", MPIRank, sharedSize, counter);

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
          // evaluate_gpu(jobs, lb, numBounds, nbBlocks, &best_l, lbound1_d, lbound2_d, parents_d, bounds_d);
          cudaDeviceSynchronize();
          endKernelCall = omp_get_wtime();
          timeLocalKernelCall[gpuID] += endKernelCall - startKernelCall;
          cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);

          /*
            Each task generates and inserts its children nodes to the pool.
          */
          // generate_children(parents, poolSize, jobs, bounds, &tree, &sol, &best_l, pool_loc);
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
              SinglePool_ext *victim;
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
              continue;
              // break;
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

    // printf("From Proc[%d] Thread[%d] OMP Critical reached\n", MPIRank, gpuID);

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

      deleteSinglePool_ext(pool_loc);
    }

  } // End of parallel region OpenMP

  MPI_Barrier(MPI_COMM_WORLD);
  // printf("From Proc[%d] Left OMP region\n", MPIRank);

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

  // for (int i = 0; i < D; i++)
  //   printf("Proc[%d] timeLocalKernelCall[%d] = %f\n", MPIRank, i, timeLocalKernelCall[i]);

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
  MPI_Reduce(&eachLocaleBest, best, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  if (MPIRank == 0)
  {
    *exploredTree += midExploredTree;
    *exploredSol += midExploredSol;
  }

  // Gather data from all processes for printing GPU workload statistics
  unsigned long long int *allExploredTrees = NULL;
  unsigned long long int *allExploredSols = NULL;
  unsigned long long int *allEachExploredTrees = NULL; // For eachExploredTree array
  unsigned long long int *allStealsProc = NULL; // For eachExploredTree array
  double *allMaxKernelCall = NULL;
  double *allMaxIdleDevice = NULL;
  if (MPIRank == 0)
  {
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

  // DEBUGGING
  // printf("From Proc[%d] Before Step 3\n", MPIRank);

  /*
    Step 3: We complete the depth-first search on CPU.
  */
  double t3;
  if (MPIRank == 0)
  {
    // int count = 0;
    startTime = omp_get_wtime();
    while (1)
    {
      int hasWork = 0;
      Node parent = popBack(&pool, &hasWork);
      if (!hasWork)
        break;
      decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
      // count++;
    }
    endTime = omp_get_wtime();
    t3 = endTime - startTime;
    *elapsedTime = t1 + t2 + t3;
  }
  // freeing memory for structs common to all MPI processes
  deleteSinglePool_ext(&pool);
  deleteSinglePool_ext(&pool_lloc);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);

  if (MPIRank == 0)
  {
    free(recvcounts);
    free(displs);
    free(masterNodes);

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

  int inst, lb, ub, m, M, D;
  double perc;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &D, &perc);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  if (MPIRank == 0)
    print_settings(inst, machines, jobs, ub, lb, D, commSize);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;

  unsigned long long int expTreeProc[commSize], expSolProc[commSize], nStealsProc[commSize];
  double timeKernelCall[commSize], timeIdle[commSize], workloadProc[commSize];

  for (int i = 0; i < commSize; i++)
  {
    expTreeProc[i] = 0;
    expSolProc[i] = 0;
    timeKernelCall[i] = 0;
    timeIdle[i] = 0;
    workloadProc[i] = 0;
    nStealsProc[i] = 0;
  }

  pfsp_search(inst, lb, m, M, D, perc, &optimum, &exploredTree, &exploredSol, &elapsedTime, expTreeProc, expSolProc, nStealsProc, timeKernelCall, timeIdle, workloadProc, MPIRank, commSize);

  if (MPIRank == 0)
  {
    print_results(optimum, exploredTree, exploredSol, elapsedTime);
    print_results_file(inst, machines, jobs, lb, D, commSize, optimum, exploredTree, exploredSol, elapsedTime, expTreeProc, expSolProc, nStealsProc, timeKernelCall, timeIdle, workloadProc);
  }

  MPI_Finalize();

  return 0;
}