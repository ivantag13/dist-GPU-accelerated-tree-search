#include "Pool_atom.h"

// Initialization of pool.
void initSinglePool_atom(SinglePool_atom *pool)
{
  pool->elements = (Node *)malloc(INITIAL_CAPACITY * sizeof(Node));
  pool->capacity = INITIAL_CAPACITY;
  pool->front = 0;
  pool->size = 0;
  atomic_store(&(pool->lock), false);
}

// Static cyclical distribution of nodes from pool_source to pool. Parallel safety is not guaranteed.
void roundRobin_distribution(SinglePool_atom *pool, SinglePool_atom *pool_source, int poolID, int step)
{
  const int poolSize = pool_source->size;
  const int c = poolSize / step;
  const int l = poolSize - (step - 1) * c;
  const int f = pool_source->front;

  // each task gets its chunk
  for (int i = 0; i < c; i++)
  {
    pool->elements[i] = pool_source->elements[poolID + f + i * step];
  }
  pool->size += c;
  if (poolID == step - 1)
  {
    for (int i = c; i < l; i++)
    {
      pool->elements[i] = pool_source->elements[(step * c) + f + i - c];
    }
    pool->size += l - c;
  }
  return;
}

// Parallel-safe insertion to the end of the deque.
void pushBack(SinglePool_atom *pool, Node node)
{
  bool expected;
  while (true)
  {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true))
    {
      if (pool->front + pool->size >= pool->capacity)
      {
        pool->capacity *= 2;
        pool->elements = realloc(pool->elements, pool->capacity * sizeof(Node));
      }

      pool->elements[pool->front + pool->size] = node;
      pool->size += 1;
      atomic_store(&(pool->lock), false);
      return;
    }
  }
}

// Insertion to the end of the deque. Parallel safety is not guaranteed.
void pushBackFree(SinglePool_atom *pool, Node node)
{
  if (pool->front + pool->size >= pool->capacity)
  {
    pool->capacity *= 2;
    pool->elements = (Node *)realloc(pool->elements, pool->capacity * sizeof(Node));
  }

  pool->elements[pool->front + pool->size] = node;
  pool->size += 1;
}

// Parallel-safe bulk insertion to the end of the deque.
void pushBackBulk(SinglePool_atom *pool, Node *nodes, int size)
{
  bool expected;
  while (true)
  {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true))
    {
      if (pool->front + pool->size + size >= pool->capacity)
      {
        pool->capacity *= pow(2, ceil(log2((double)(pool->front + pool->size + size) / pool->capacity)));
        pool->elements = realloc(pool->elements, pool->capacity * sizeof(Node));
      }

      for (int i = 0; i < size; i++)
        pool->elements[pool->front + pool->size + i] = nodes[i];
      pool->size += size;
      atomic_store(&(pool->lock), false);
      return;
    }
  }
}
// Parallel-safe bulk insertion to the end of the deque. Parallel safety is not guaranteed.
void pushBackBulkFree(SinglePool_atom *pool, Node *nodes, int size)
{
  if (pool->front + pool->size + size >= pool->capacity)
  {
    pool->capacity *= pow(2, ceil(log2((double)(pool->front + pool->size + size) / pool->capacity)));
    pool->elements = realloc(pool->elements, pool->capacity * sizeof(Node));
  }

  for (int i = 0; i < size; i++)
    pool->elements[pool->front + pool->size + i] = nodes[i];
  pool->size += size;
  return;
}

// Parallel-safe removal from the end of the deque.
Node popBack(SinglePool_atom *pool, int *hasWork)
{
  bool expected;
  while (true)
  {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, false))
    {
      if (pool->size > 0)
      {
        *hasWork = 1;
        pool->size -= 1;
        Node elt = pool->elements[pool->front + pool->size];
        atomic_store(&pool->lock, false);
        return elt;
      }
      else
      {
        atomic_store(&(pool->lock), false);
        break;
      }
    }
  }

  return (Node){0};
}

// Removal from the end of the deque. Parallel safety is not guaranteed.
Node popBackFree(SinglePool_atom *pool, int *hasWork)
{
  if (pool->size > 0)
  {
    *hasWork = 1;
    pool->size -= 1;
    return pool->elements[pool->front + pool->size];
  }

  return (Node){0};
}

// Parallel-safe bulk removal from the end of the deque.
int popBackBulk(SinglePool_atom *pool, const int m, const int M, Node *parents)
{
  bool expected;
  while (true)
  {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true))
    {
      if (pool->size < m)
      {
        atomic_store(&(pool->lock), false);
        return pool->size;
      }
      else
      {
        int poolSize = MIN(pool->size, M);
        pool->size -= poolSize;
        for (int i = 0; i < poolSize; i++)
          parents[i] = pool->elements[pool->front + pool->size + i];
        atomic_store(&(pool->lock), false);
        return poolSize;
      }
    }
  }
}

// Parallel-safe bulk removal from the end of the deque based on the steal-half strategy.
int popBackBulkHalf(SinglePool_atom *pool, const int m, const int M, Node *parents)
{
  bool expected;
  while (true)
  {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true))
    {
      if (pool->size < 2 * m)
      {
        atomic_store(&(pool->lock), false);
        return pool->size;
      }
      else
      {
        int poolSize = MIN(pool->size / 2, M);
        pool->size -= poolSize;
        for (int i = 0; i < poolSize; i++)
          parents[i] = pool->elements[pool->front + pool->size + i];
        atomic_store(&(pool->lock), false);
        return poolSize;
      }
    }
  }
}

// Bulk removal from the end of the deque. Parallel safety is not guaranteed.
int popBackBulkFree(SinglePool_atom *pool, const int m, const int M, Node *parents)
{
  if (pool->size >= m)
  {
    int poolSize = MIN(pool->size, M);
    pool->size -= poolSize;
    for (int i = 0; i < poolSize; i++)
    {
      parents[i] = pool->elements[pool->front + pool->size + i];
    }
    return poolSize;
  }
  return pool->size;
}

// Bulk removal from the end of the deque based on the steal-half strategy. Parallel safety is not guaranteed.
int popBackBulkHalfFree(SinglePool_atom *pool, const int m, const int M, Node *parents)
{
  if (pool->size < 2 * m){
    printf("DEADCODE\n");
    exit(-1);
    return pool->size;
  }
  else
  {
    int poolSize = MIN(pool->size / 2, M);
    pool->size -= poolSize;
    for (int i = 0; i < poolSize; i++)
      parents[i] = pool->elements[pool->front + pool->size + i];
    return poolSize;
  }
}

// Removal from the front of the deque. Parallel safety is not guaranteed.
Node popFrontFree(SinglePool_atom *pool, int *hasWork)
{
  if (pool->size > 0)
  {
    *hasWork = 1;
    pool->size--;
    return pool->elements[pool->front++];
  }

  return (Node){0};
}

// Bulk removal from the front of the deque. Parallel safety is not guaranteed.
Node *popFrontBulkFree(SinglePool_atom *pool, const int m, const int M, int *poolSize, double perc)
{
  if (pool->size >= 2 * m)
  {
    *poolSize = pool->size * perc;
    pool->size -= *poolSize;
    Node *parents = (Node *)malloc(*poolSize * sizeof(Node));
    for (int i = 0; i < *poolSize; i++)
      parents[i] = pool->elements[pool->front + i];
    pool->front += *poolSize;
    return parents;
  }

  *poolSize = 0;
  return NULL;
}

// TODO : In order to implement this function I would have to introduce a new variable
// inside struct Pool_ext (e.g. back) to keep track of the good indexes and pool size
/*Node* popHalfFrontHalfBackBulkFree(SinglePool_atom* pool, const int m, const int M, int* poolSize){
  if(pool->size >= 2*m) {
    *poolSize = pool->size/2;
    int index = *poolSize/2;
    pool->size -= (*poolSize-index);
    Node* parents = (Node*)malloc(*poolSize * sizeof(Node));
    // Steal a quarter of the work from the front
    for(int i = 0; i < index; i++)
      parents[i] = pool->elements[pool->front + i];
    pool->front += index;
    //Steal a quarter of the work from the back
    for(int i = index; i < *poolSize; i++)
      parents[i] = pool->elements[pool->front + pool->size+i];
    return parents;
  }else{
    *poolSize = 0;
    printf("\nDEADCODE\n");
    return NULL;
  }
  Node* parents = NULL;
  *poolSize = 0;
  return parents;
  }*/

// Free the memory.
void deleteSinglePool_atom(SinglePool_atom *pool)
{
  free(pool->elements);
  pool->elements = NULL;
}
