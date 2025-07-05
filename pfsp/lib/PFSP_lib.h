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

#define BLOCK_SIZE 512

#define MAX_JOBS 20
#define MAX_MACHINES 20

  void decompose_lb1(const int jobs, const lb1_bound_data *const lbound1, const Node parent,
                     int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_atom *pool);

  void decompose_lb1_d(const int jobs, const lb1_bound_data *const lbound1, const Node parent,
                       int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_atom *pool);

  void decompose_lb2(const int jobs, const lb1_bound_data *const lbound1, const lb2_bound_data *const lbound2,
                     const Node parent, int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol,
                     SinglePool_atom *pool);

  void decompose(const int jobs, const int lb, int *best,
                 const lb1_bound_data *const lbound1, const lb2_bound_data *const lbound2, const Node parent,
                 unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_atom *pool);

  void generate_children(Node *parents, Node *children, const int size, const int jobs, int *bounds, unsigned long long int *exploredTree,
                         unsigned long long int *exploredSol, int *best, SinglePool_atom *pool, int *index);

  void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb, const int D, int ws, const int commSize, const int LB, const int version);

  void print_results(const int optimum, const unsigned long long int exploredTree,
                     const unsigned long long int exploredSol, const double timer);

  void parse_parameters(int argc, char *argv[], int *inst, int *lb, int *ub, int *m, int *M, int *D, int *ws, int *L, double *perc);

#ifdef __cplusplus
}
#endif

#endif // PFSP_LIB_H
