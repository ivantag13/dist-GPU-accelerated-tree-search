#ifdef __cplusplus
extern "C"
{
#endif

#include <stdlib.h>
#include "PFSP_gpu_lib.cuh"
#include "bounds_gpu.cu"

  // CUDA error checking
  // TODO: fix portability for variable cudaError_t (https://rocm.docs.amd.com/projects/HIP/en/docs-develop/how-to/hip_porting_guide.html)
  // #define gpuErrchk(ans)                          \
  //   {                                             \
  //     gpuAssert((ans), __FILE__, __LINE__, true); \
  //   }
  //   void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
  //   {
  //     if (code != cudaSuccess)
  //     {
  //       fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  //       if (abort)
  //         exit(code);
  //     }
  //   }

  __device__ void swap_cuda(int *a, int *b)
  {
    int tmp = *b;
    *b = *a;
    *a = tmp;
  }

  void printDims(dim3 gridDim, dim3 blockDim)
  {
    printf("Grid Dimensions : [%d, %d, %d] blocks. \n",
           gridDim.x, gridDim.y, gridDim.z);

    printf("Block Dimensions : [%d, %d, %d] threads.\n",
           blockDim.x, blockDim.y, blockDim.z);
  }

  // Evaluate a bulk of parent nodes on GPU using lb1
  __global__ void evaluate_gpu_lb1(const int jobs, const int size, Node *parents_d, const int parentsSize,
                                   const lb1_bound_data lbound1_d, int *bounds, int *sumOffSets_d, int *nodeIndex)
  {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < size)
    {
      int parentId = nodeIndex[threadId];
      Node parent = parents_d[parentId];
      int depth = parent.depth;
      int limit1 = parent.limit1;
      int prmu[MAX_JOBS];
      for (int i = 0; i < MAX_JOBS; i++)
        prmu[i] = parent.prmu[i];
      int k = threadId + depth;
      if (parentId != 0)
        k -= sumOffSets_d[parentId - 1];

      swap_cuda(&prmu[depth], &prmu[k]);
      lb1_bound_gpu(lbound1_d, prmu, limit1 + 1, jobs, &bounds[threadId]);
      // swap_cuda(&parent.prmu[depth], &parent.prmu[k]);
    }
  }

  /*
    NOTE: This lower bound evaluates all the children of a given parent at the same time.
    Therefore, the GPU loop is on the parent nodes and not on the children ones, in contrast
    to the other lower bounds.
  */
  // Evaluate a bulk of parent nodes on GPU using lb1_d.
  __global__ void evaluate_gpu_lb1_d(const int jobs, const int size, Node *parents_d, const lb1_bound_data lbound1_d, int *bounds, int *sumOffSets_d)
  {
    int parentId = blockIdx.x * blockDim.x + threadIdx.x;

    if (parentId < size)
    {
      Node parent = parents_d[parentId];

      // Vector of integers of size MAX_JOBS
      int lb_begin[MAX_JOBS];
      int limit1 = parent.limit1;
      int prmu[MAX_JOBS];
      for (int i = 0; i < MAX_JOBS; i++)
        prmu[i] = parent.prmu[i];

      lb1_children_bounds_gpu(lbound1_d, prmu, limit1, jobs, lb_begin);

      // for (int k = 0; k < jobs; k++)
      // {
      for (int k = limit1 + 1; k < jobs; k++)
      {
        const int job = prmu[k];
        int index = (k - (limit1 + 1));
        if (parentId != 0)
          index += sumOffSets_d[parentId - 1];
        bounds[index] = lb_begin[job];
      }
      // }
    }
  }

  // Evaluate a bulk of parent nodes on GPU using lb2.
  __global__ void evaluate_gpu_lb2(const int jobs, const int size, int best, Node *parents_d, int parentsSize, const lb1_bound_data lbound1_d,
                                   const lb2_bound_data lbound2_d, int *bounds, int *sumOffSets_d, int *nodeIndex_d)
  {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < size)
    {
      int parentId = nodeIndex_d[threadId];
      Node parent = parents_d[parentId];
      int depth = parent.depth;
      int limit1 = parent.limit1;
      int prmu[MAX_JOBS];
      for (int i = 0; i < MAX_JOBS; i++)
        prmu[i] = parent.prmu[i];
      int k = threadId + depth;
      if (parentId != 0)
        k -= sumOffSets_d[parentId - 1];

      swap_cuda(&prmu[depth], &prmu[k]);
      lb2_bound_gpu(lbound1_d, lbound2_d, prmu, limit1 + 1, jobs, best, &bounds[threadId]);
      // swap_cuda(&parent.prmu[depth], &parent.prmu[k]);
    }
  }

  void evaluate_gpu(const int jobs, const int lb, const int size, const int nbBlocks, const int parentsSize, int *best,
                    const lb1_bound_data lbound1, const lb2_bound_data lbound2, Node *parents, int *bounds, int *sumOffSets_d, int *nodeIndex_d)
  {
    // 1D grid of 1D nbBlocks(_lb1_d) blocks with block size BLOCK_SIZE
    int nbBlocks_lb1_d;
    switch (lb)
    {
    case 0: // lb1_d
      nbBlocks_lb1_d = ceil((double)parentsSize / BLOCK_SIZE);
      evaluate_gpu_lb1_d<<<nbBlocks_lb1_d, BLOCK_SIZE>>>(jobs, parentsSize, parents, lbound1, bounds, sumOffSets_d);
      return;
      break;

    case 1: // lb1
      evaluate_gpu_lb1<<<nbBlocks, BLOCK_SIZE>>>(jobs, size, parents, parentsSize, lbound1, bounds, sumOffSets_d, nodeIndex_d);
      return;
      break;

    case 2: // lb2
      evaluate_gpu_lb2<<<nbBlocks, BLOCK_SIZE>>>(jobs, size, *best, parents, parentsSize, lbound1, lbound2, bounds, sumOffSets_d, nodeIndex_d);
      return;
      break;
    }
  }

  void lb1_alloc_gpu(lb1_bound_data *lbound1_d, lb1_bound_data *lbound1, int *p_times_d, int *min_heads_d, int *min_tails_d, int jobs, int machines)
  {
    // Allocating and copying memory necessary for deep copy of lbound1
    cudaMalloc((void **)&p_times_d, jobs * machines * sizeof(int));
    cudaMalloc((void **)&min_heads_d, machines * sizeof(int));
    cudaMalloc((void **)&min_tails_d, machines * sizeof(int));
    cudaMemcpy(p_times_d, lbound1->p_times, (jobs * machines) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(min_heads_d, lbound1->min_heads, machines * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(min_tails_d, lbound1->min_tails, machines * sizeof(int), cudaMemcpyHostToDevice);

    // Deep copy of lbound1
    lbound1_d->p_times = p_times_d;
    lbound1_d->min_heads = min_heads_d;
    lbound1_d->min_tails = min_tails_d;
    lbound1_d->nb_jobs = lbound1->nb_jobs;
    lbound1_d->nb_machines = lbound1->nb_machines;

    return;
  }

  void lb2_alloc_gpu(lb2_bound_data *lbound2_d, lb2_bound_data *lbound2, int *johnson_schedule_d, int *lags_d,
                     int *machine_pairs_1_d, int *machine_pairs_2_d, int *machine_pair_order_d, int jobs, int machines)
  {
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
    lbound2_d->johnson_schedules = johnson_schedule_d;
    lbound2_d->lags = lags_d;
    lbound2_d->machine_pairs_1 = machine_pairs_1_d;
    lbound2_d->machine_pairs_2 = machine_pairs_2_d;
    lbound2_d->machine_pair_order = machine_pair_order_d;
    lbound2_d->nb_machine_pairs = lbound2->nb_machine_pairs;
    lbound2_d->nb_jobs = lbound2->nb_jobs;
    lbound2_d->nb_machines = lbound2->nb_machines;
    return;
  }

#ifdef __cplusplus
}
#endif

/*******************************************************************************
FLOP estimation
*******************************************************************************/
// TODO: (TO FIX) PFLOPs estimation on the GPU
// TODO: Fix FLOP estimations or use appropriate tool to measure it
// Number of machine‑pairs as a function of M
//   P = M*(M‑1)/2
int P_of(int M)
{
  return (int)M * (M - 1) / 2;
}

// FLOP count per single lb1 bound invocation,
// as a function of N (jobs), M (machines), and lim (limit1):
int flop_lb1(int N, int M, int lim)
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
int bytes_per_inv_lb1(int N, int M)
{
  int loads = (int)N * M;
  int stores = 1;
  return 4 * (loads + stores);
}

// FLOP count per single lb2 bound invocation:
//   same front/remain cost as lb1, plus the lb_makespan loop over P pairs
int flop_lb2(int N, int M, int lim)
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
int bytes_per_inv_lb2(int N, int M, int lim)
{
  int P = P_of(M);
  int uns = (int)N - lim;
  int loads = (int)N * M + 3 * P * uns;
  int stores = 1;
  return 4 * (loads + stores);
}