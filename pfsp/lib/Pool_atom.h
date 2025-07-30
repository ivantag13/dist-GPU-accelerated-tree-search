#ifndef POOL_ATOM_H
#define POOL_ATOM_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "PFSP_node.h"
#include <stdbool.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <math.h>

  /*******************************************************************************
  Extension of the "Pool" data structure ensuring parallel-safety and supporting
  bulk operations.
  *******************************************************************************/

#define INITIAL_CAPACITY 1024

  typedef struct
  {
    Node *elements;
    int capacity;
    int front;
    int size;
    _Atomic bool lock;
  } SinglePool_atom;

  void initSinglePool_atom(SinglePool_atom *pool);

  void roundRobin_distribution(SinglePool_atom *pool, SinglePool_atom *pool_source, int poolID, int step);

  void pushBack(SinglePool_atom *pool, Node node);

  void pushBackFree(SinglePool_atom *pool, Node node);

  void pushBackBulk(SinglePool_atom *pool, Node *nodes, int size);

  void pushBackBulkFree(SinglePool_atom *pool, Node *nodes, int size);

  Node popBack(SinglePool_atom *pool, int *hasWork);

  Node popBackFree(SinglePool_atom *pool, int *hasWork);

  int popBackBulk(SinglePool_atom *pool, const int m, const int M, Node *parents);

  int popBackBulkFree(SinglePool_atom *pool, const int m, const int M, Node *parents);

  Node *popBackBulkHalf(SinglePool_atom *pool, const int m, const int M, int *Half);

  Node popFrontFree(SinglePool_atom *pool, int *hasWork);

  Node *popFrontBulkFree(SinglePool_atom *pool, const int m, const int M, int *poolSize, double perc);

  // Node* popHalfFrontHalfBackBulkFree(SinglePool_atom* pool, const int m, const int M, int* poolSize);

  void deleteSinglePool_atom(SinglePool_atom *pool);

#ifdef __cplusplus
}
#endif

#endif // POOL_ATOM_H
