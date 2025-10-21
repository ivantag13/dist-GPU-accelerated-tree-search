/*******************************************************************************
Implementation of PFSP Nodes.
*******************************************************************************/
#include "PFSP_lib.h"

// Evaluate and generate children nodes on CPU using LB1 bounding function. Parallel safety is not guaranteed.
inline void decompose_lb1(const int jobs, const lb1_bound_data *const lbound1, const Node parent,
                          int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_atom *pool)
{
  for (int i = parent.limit1 + 1; i < jobs; i++)
  {
    Node child;
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int16_t));
    swap(&child.prmu[parent.depth], &child.prmu[i]);
    int limit1 = child.limit1;
    int prmu[MAX_JOBS];
    for (int j = 0; j < MAX_JOBS; j++)
      prmu[j] = child.prmu[j];

    int lowerbound = lb1_bound(lbound1, prmu, limit1, jobs);

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
        pushBackFree(pool, child);
        *tree_loc += 1;
      }
    }
  }
}

// Evaluate and generate children nodes on CPU using LB1_d bounding function. Parallel safety is not guaranteed.
inline void decompose_lb1_d(const int jobs, const lb1_bound_data *const lbound1, const Node parent,
                            int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol, SinglePool_atom *pool)
{
  // This declaration is better for performance, memory is on stack
  int lb_begin[jobs];
  // int *lb_begin = (int *)malloc(jobs * sizeof(int));

  int limit1 = parent.limit1;
  int prmu[MAX_JOBS];
  for (int j = 0; j < MAX_JOBS; j++)
    prmu[j] = parent.prmu[j];

  lb1_children_bounds(lbound1, prmu, limit1, jobs, lb_begin);

  for (int i = parent.limit1 + 1; i < jobs; i++)
  {
    Node child;
    int job = parent.prmu[i];
    int lb = lb_begin[job];

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
        child.depth = parent.depth + 1;
        child.limit1 = parent.limit1 + 1;
        memcpy(child.prmu, parent.prmu, jobs * sizeof(int16_t));
        swap(&child.prmu[parent.depth], &child.prmu[i]);

        pushBackFree(pool, child);
        *tree_loc += 1;
      }
    }
  }

  // free(lb_begin);
}

// Evaluate and generate children nodes on CPU using LB2 bounding function. Parallel safety is not guaranteed.
inline void decompose_lb2(const int jobs, const lb1_bound_data *const lbound1, const lb2_bound_data *const lbound2,
                          const Node parent, int *best, unsigned long long int *tree_loc, unsigned long long int *num_sol,
                          SinglePool_atom *pool)
{
  for (int i = parent.limit1 + 1; i < jobs; i++)
  {
    Node child;
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int16_t));
    swap(&child.prmu[parent.depth], &child.prmu[i]);
    int limit1 = child.limit1;
    int prmu[MAX_JOBS];
    for (int j = 0; j < MAX_JOBS; j++)
      prmu[j] = child.prmu[j];

    int lowerbound = lb2_bound(lbound1, lbound2, prmu, limit1, jobs, *best);

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
        pushBackFree(pool, child);
        *tree_loc += 1;
      }
    }
  }
}

// Printing functions

void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb, const int D, const int C, int ws, const int commSize, const int LB, const int version)
{
  printf("\n=================================================\n");
  if (version == 0)
    printf("Sequential C\n\n");
  else if (version == 1)
    printf("Single-GPU C+CUDA\n\n");
  else if (version == 2)
    printf("Multi-core Multi-GPU C+OpenMP+CUDA (%d GPU(s) / %d CPU(s) - [%d]WS)\n\n", D, C, ws);
  else
    printf("Distributed Multi-GPU C+MPI+OpenMP+CUDA (%d MPI processes x ( %d GPU(s) / %d CPU(s) ) - LB[%d])\n\n", commSize, D, C, LB);

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

// Setting parameters function
void parse_parameters(int argc, char *argv[], int *inst, int *lb, int *ub, int *m, int *M, int *T, int *D, int *C, int *ws, int *L, double *perc)
{
  *inst = 14;
  *lb = 1;
  *ub = 1;
  *m = 25;
  *M = 50000;
  *T = 5000;
  *D = 0;
  *C = 0;
  *ws = 1;
  *L = 1;
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
      {"T", required_argument, NULL, 'T'},
      {"D", required_argument, NULL, 'D'},
      {"C", required_argument, NULL, 'C'},
      {"ws", required_argument, NULL, 'w'},
      {"L", required_argument, NULL, 'L'},
      {"perc", required_argument, NULL, 'p'},
      {NULL, 0, NULL, 0} // Terminate options array
  };

  int opt, value;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "i:l:u:m:M:T:D:C:w:L:p:", long_options, &option_index)) != -1)
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

    case 'T':
      if (value < *m)
      {
        fprintf(stderr, "Error: unsupported maximal pool for CPU multi-core\n");
        exit(EXIT_FAILURE);
      }
      *T = value;
      break;

    case 'D':
      if (value < 0)
      {
        fprintf(stderr, "Error: unsupported number of GPU(s)\n");
        exit(EXIT_FAILURE);
      }
      *D = value;
      break;

    case 'C':
      if (value < 0)
      {
        fprintf(stderr, "Error: unsupported number of CPU Core(s)\n");
        exit(EXIT_FAILURE);
      }
      *C = value;
      break;

    case 'w':
      if (value < 0 || value > 1)
      {
        fprintf(stderr, "Error: unsupported Intra-node Work Stealing option\n");
        exit(EXIT_FAILURE);
      }
      *ws = value;
      break;

    case 'L':
      if (value < 0 || value > 1)
      {
        fprintf(stderr, "Error: unsupported distributed dynamic load balancing option\n");
        exit(EXIT_FAILURE);
      }
      *L = value;
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
      fprintf(stderr, "Usage: %s --inst <value> --lb <value> --ub <value> --m <value> --M <value> --T <value> --D <value> --C <value> --w <value> --L <value> --perc <value>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}
