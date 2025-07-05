#ifndef EVALUATE_H
#define EVALUATE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "PFSP_node.h"		 // For Nodes definition
#include "c_bound_simple.h"	 // For structs definitions
#include "c_bound_johnson.h" // For structs definitions

	void gpuAssert(cudaError_t code, const char *file, int line, bool abort);

	void evaluate_gpu(const int jobs, const int lb, const int size, const int nbBlocks, const int parentsSize,
					  int *best, const lb1_bound_data lbound1, const lb2_bound_data lbound2, Node *parents, int *bounds, int *sumOffSets_d, int *nodeIndex_d);

	void lb1_alloc_gpu(lb1_bound_data *lbound1_d, lb1_bound_data *lbound1, int *p_times_d, int *min_heads_d, int *min_tails_d, int jobs, int machines);

	void lb2_alloc_gpu(lb2_bound_data *lbound2_d, lb2_bound_data *lbound2, int *johnson_schedule_d, int *lags_d,
					   int *machine_pairs_1_d, int *machine_pairs_2_d, int *machine_pair_order_d, int jobs, int machines);

#ifdef __cplusplus
}
#endif

#endif // EVALUATE_H
