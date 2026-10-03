#ifndef GPU_HIERARCHICAL_REDUCTION_H
#define GPU_HIERARCHICAL_REDUCTION_H

#include <cstddef>

// All pointers are CUDA device pointers. Compute sum(input[i] * input[i]).
void HierarchicalReduction(std::size_t n, const float *input, float *result,
                           float *workspace);

#endif
