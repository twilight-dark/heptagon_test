#include "reduction.h"
#include <cuda_runtime.h>

namespace {
constexpr unsigned int kThreads = 256;
constexpr unsigned int kItemsPerThread = 4;
constexpr unsigned int kChunk = kThreads * kItemsPerThread;
constexpr unsigned int kMaxBlocks = 1024;
constexpr std::size_t kSingleBlockLimit = 4096;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <bool Square>
__global__ void reduce_kernel(std::size_t n, const float *__restrict__ input,
                              float *__restrict__ output) {
    __shared__ float warp_sums[kThreads / 32];
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * kChunk;
    float sums[kItemsPerThread] = {};

    // Adjacent lanes read adjacent elements; independent accumulators shorten
    // the dependency chains. Bounds checks cover arbitrary input lengths.
    for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * kChunk;
         base < n; base += step) {
#pragma unroll
        for (unsigned int item = 0; item < kItemsPerThread; ++item) {
            const std::size_t index = base + tid + item * kThreads;
            if (index < n) {
                const float value = input[index];
                if constexpr (Square) {
                    sums[item] += value * value;
                } else {
                    sums[item] += value;
                }
            }
        }
    }

    float sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    sum = warp_sum(sum);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    // All lanes of warp zero participate, including lanes with no partial sum.
    if (warp == 0) {
        sum = lane < kThreads / 32 ? warp_sums[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) output[blockIdx.x] = sum;
    }
}
}  // namespace

void HierarchicalReduction(std::size_t n, const float *input, float *result, float *workspace) {
    if (n <= kSingleBlockLimit) {
        reduce_kernel<true><<<1, kThreads>>>(n, input, result);
        return;
    }

    const std::size_t chunks = n / kChunk + (n % kChunk != 0);
    const unsigned int blocks = static_cast<unsigned int>(
        chunks < kMaxBlocks ? chunks : kMaxBlocks);
    reduce_kernel<true><<<blocks, kThreads>>>(n, input, workspace);
    // Same-stream ordering makes all partial sums visible without a host sync.
    reduce_kernel<false><<<1, kThreads>>>(blocks, workspace, result);
}
