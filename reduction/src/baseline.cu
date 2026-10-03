#include <cuda_runtime.h>
#include "reduction.h"

namespace {
__global__ void atomic_sum(std::size_t n, const float *input, float *result) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) atomicAdd(result, input[index] * input[index]);
}
}  // namespace

void BaselineReduction(std::size_t n, const float *input, float *result, float *workspace) {
    (void)workspace;
    cudaMemsetAsync(result, 0, sizeof(float));
    if (n != 0) atomic_sum<<<static_cast<unsigned int>((n + 255) / 256), 256>>>(n, input, result);
}
