#ifndef PFSP_NODE_H
#define PFSP_NODE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdlib.h>
#include <stdint.h>
#include "c_bound_simple.h"	 // For structs definitions
#include "c_bound_johnson.h" // For structs definitions
#include "macro.h"

    typedef struct
    {
        int16_t depth;
        int16_t limit1;
        int16_t prmu[MAX_JOBS];
    } Node;

    void initRoot(Node *root, const int jobs);

    int compare_nodes(const void *a, const void *b);

#ifdef __cplusplus
}
#endif

#endif // PFSP_NODE_H