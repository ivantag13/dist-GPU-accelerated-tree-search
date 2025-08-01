#ifndef PFSP_LIB_H
#define PFSP_LIB_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include "../../common/util.h"
#include "c_bound_simple.h"
#include "c_bound_johnson.h"
#include "Pool_atom.h"
#include "macro.h"

  void decompose_lb1(const int jobs, const lb1_bound_data *const lbound1, const Node parent,
                     int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_atom *pool);

  void decompose_lb1_d(const int jobs, const lb1_bound_data *const lbound1, const Node parent,
                       int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_atom *pool);

  void decompose_lb2(const int jobs, const lb1_bound_data *const lbound1, const lb2_bound_data *const lbound2,
                     const Node parent, int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol,
                     SinglePool_atom *pool);

  static inline void decompose(const int jobs, const int lb, int *best,
                               const lb1_bound_data *const lbound1, const lb2_bound_data *const lbound2, const Node parent,
                               unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_atom *pool)
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
  static inline void generate_children(Node *parents, Node *children, const int size, const int jobs, int *bounds, unsigned long long int *exploredTree,
                                       unsigned long long int *exploredSol, int *best, SinglePool_atom *pool, int *index)
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
        const int lowerbound = bounds[(j - (limit1 + 1)) + sum];

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

            *exploredTree += 1;
          }
        }
      }
      sum += jobs - depth;
    }
    *index = childrenIndex;
  }

  void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb, const int D, int ws, const int commSize, const int LB, const int version);

  void print_results(const int optimum, const unsigned long long int exploredTree,
                     const unsigned long long int exploredSol, const double timer);

  void parse_parameters(int argc, char *argv[], int *inst, int *lb, int *ub, int *m, int *M, int *D, int *ws, int *L, double *perc);

#ifdef __cplusplus
}
#endif

#endif // PFSP_LIB_H
