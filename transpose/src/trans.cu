#include "trans.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace {
constexpr int kTile = 32;
constexpr int kBlockRows = 16;

__global__ void transpose_and_mix(int N, const double *__restrict__ A,
                                  double *__restrict__ B) {
    __shared__ double tile[kTile][kTile + 1];
    const int lane = threadIdx.x;
    const int row = threadIdx.y;
    const std::size_t stride = static_cast<std::size_t>(N);
    const std::size_t input_col = blockIdx.x * kTile + lane;
    const std::size_t input_row = blockIdx.y * kTile;

    // Each warp reads contiguous input elements. N is a multiple of 32.
#pragma unroll
    for (int r = row; r < kTile; r += kBlockRows) {
        tile[r][lane] = A[(input_row + r) * stride + input_col];
    }
    __syncthreads();

    const std::size_t output_row = blockIdx.x * kTile;
    const std::size_t output_col = blockIdx.y * kTile + lane;
#pragma unroll
    for (int r = row; r < kTile; r += kBlockRows) {
        // After transposition a warp owns one complete 32-element group.
        double value = tile[lane][r];
#pragma unroll
        for (int round = 0; round < 8; ++round) {
            // Exchange old values before updating, including lane 31 -> 0.
            const double next = __shfl_sync(0xffffffffu, value,
                                            (lane + 1) & 31);
            value += next;
        }
        B[(output_row + r) * stride + output_col] = value;
    }
}
}  // namespace

void MatrixTranspose(int N, const double *A, double *B) {
    if (N <= 0) return;
    const dim3 block(kTile, kBlockRows);
    const dim3 grid(N / kTile, N / kTile);
    transpose_and_mix<<<grid, block>>>(N, A, B);
}
