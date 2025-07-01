#ifdef __cplusplus
extern "C"
{
#endif

#include <stdlib.h>
#include "evaluate.h"
#include "bounds_gpu.cu"

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

      // if (threadId >= sumOffSets_d[parentId])
      // {
      //   while (parentId < parentsSize - 1)
      //   {
      //     if (threadId >= sumOffSets_d[parentId] && threadId < sumOffSets_d[++parentId])
      //       break;
      //   }
      // }

      // printf("Thread[%d] parentdId[%d] nodeIndex[%d]\n", threadId, parentId, nodeIndex[threadId]);

      Node parent = parents_d[parentId];
      int depth = parent.depth;
      int limit1 = parent.limit1;
      int k = threadId + depth;
      if (parentId != 0)
        k -= sumOffSets_d[parentId - 1];

      swap_cuda(&parent.prmu[depth], &parent.prmu[k]);
      lb1_bound_gpu(lbound1_d, parent.prmu, limit1 + 1, jobs, &bounds[threadId]);
      // swap_cuda(&parent.prmu[depth], &parent.prmu[k]);
    }
  }

  /*
    NOTE: This lower bound evaluates all the children of a given parent at the same time.
    Therefore, the GPU loop is on the parent nodes and not on the children ones, in contrast
    to the other lower bounds.
  */
  // Evaluate a bulk of parent nodes on GPU using lb1_d.
  __global__ void evaluate_gpu_lb1_d(const int jobs, const int size, Node *parents_d, const lb1_bound_data lbound1_d, int *bounds)
  {
    int parentId = blockIdx.x * blockDim.x + threadIdx.x;

    if (parentId < size)
    {
      Node parent = parents_d[parentId];

      // Vector of integers of size MAX_JOBS
      int lb_begin[MAX_JOBS];

      lb1_children_bounds_gpu(lbound1_d, parent.prmu, parent.limit1, jobs, lb_begin);

      for (int k = 0; k < jobs; k++)
      {
        if (k >= parent.limit1 + 1)
        {
          const int job = parent.prmu[k];
          bounds[parentId * jobs + k] = lb_begin[job];
        }
      }
    }
  }

  // Evaluate a bulk of parent nodes on GPU using lb2.
  __global__ void evaluate_gpu_lb2(const int jobs, const int size, int best, Node *parents_d, int parentsSize, const lb1_bound_data lbound1_d,
                                   const lb2_bound_data lbound2_d, int *bounds, int *sumOffSets_d, int *nodeIndex_d)
  {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < size)
    {
      // int index;
      // int i = 0;
      // int sizeVector = parentsSize;

      // if (threadId < sumOffSets_d[i])
      // {
      //   index = i;
      // }
      // else
      // {
      //   while (i < sizeVector - 1)
      //   {
      //     if (threadId >= sumOffSets_d[i] && threadId < sumOffSets_d[i + 1])
      //     {
      //       index = i + 1;
      //       break;
      //     }
      //     i++;
      //   }
      // }
      // // threadOffSet = offSets_d[index];
      // if (threadId >= size || index >= sizeVector)
      //   printf("Error : bad index for sizes threadId[%d], size[%d], index[%d], sizeVector[%d]\n", threadId, size, index, sizeVector);

      // // All thread indexation is done in these two next lines
      // const int parentId = index;
      // Node parent = parents_d[parentId];
      // int depth = parent.depth;
      // int limit1 = parent.limit1;
      // int k;
      // if (index == 0)
      // {
      //   k = threadId + depth; // threadId-sumOffSets_d[index] should vary from 0 to offSets_d[index]
      // }
      // else
      // {
      //   k = (threadId - sumOffSets_d[index - 1]) + depth; // threadId-sumOffSets_d[index] should vary from 0 to offSets_d[index]
      // }

      // if (k < limit1 + 1 || k >= jobs)
      //   printf("Thread[%d] Something wrong k[%d] index[%d] sumOffSets_d[%d]\n", threadId, k, index, sumOffSets_d[index]);
      // // We evaluate all permutations by varying index k from limit1 forward
      // // if (k >= limit1 + 1)
      // //{

      // int parentId;

      // if (threadId <= size / 2)
      // {
      //   parentId = 0;
      //   if (threadId >= sumOffSets_d[parentId])
      //   {
      //     while (parentId < parentsSize - 1)
      //     {
      //       if (threadId >= sumOffSets_d[parentId] && threadId < sumOffSets_d[++parentId])
      //         break;
      //     }
      //   }
      // }
      // else
      // {
      //   parentId = parentsSize - 1;
      //   while (parentId >= 0)
      //   {
      //     if (threadId < sumOffSets_d[parentId] && threadId >= sumOffSets_d[parentId - 1])
      //       break;
      //     parentId--;
      //   }
      // }

      int parentId = nodeIndex_d[threadId];
      Node parent = parents_d[parentId];
      int depth = parent.depth;
      int limit1 = parent.limit1;
      int k = threadId + depth;
      if (parentId != 0)
        k -= sumOffSets_d[parentId - 1];

      swap_cuda(&parent.prmu[depth], &parent.prmu[k]);
      lb2_bound_gpu(lbound1_d, lbound2_d, parent.prmu, limit1 + 1, jobs, best, &bounds[threadId]);
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
      nbBlocks_lb1_d = ceil((double)nbBlocks / jobs);
      evaluate_gpu_lb1_d<<<nbBlocks_lb1_d, BLOCK_SIZE>>>(jobs, size, parents, lbound1, bounds);
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
#ifdef __cplusplus
}
#endif
