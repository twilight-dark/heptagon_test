#include "trans.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace {
constexpr int kTile = 32;
constexpr int kBlockRows = 16;
constexpr int kValuesPerThread = kTile / kBlockRows;
static_assert(kTile % kBlockRows == 0, "Block rows must divide the tile size");

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
    // Keep independent output groups live together to expose instruction-level
    // parallelism across their otherwise dependent eight-round chains.
    double values[kValuesPerThread];
#pragma unroll
    for (int group = 0; group < kValuesPerThread; ++group) {
        values[group] = tile[lane][row + group * kBlockRows];
    }
#pragma unroll
    for (int round = 0; round < 8; ++round) {
        double next[kValuesPerThread];
#pragma unroll
        for (int group = 0; group < kValuesPerThread; ++group) {
            // Each group still exchanges only its own previous-round values.
            next[group] = __shfl_sync(0xffffffffu, values[group], (lane + 1) & 31);
        }
#pragma unroll
        for (int group = 0; group < kValuesPerThread; ++group) {
            values[group] += next[group];
        }
    }
#pragma unroll
    for (int group = 0; group < kValuesPerThread; ++group) {
        const int r = row + group * kBlockRows;
        B[(output_row + r) * stride + output_col] = values[group];
    }
}
}  // namespace

void MatrixTranspose(int N, const double *A, double *B) {
    if (N <= 0) return;
    const dim3 block(kTile, kBlockRows);
    const dim3 grid(N / kTile, N / kTile);
    transpose_and_mix<<<grid, block>>>(N, A, B);
}
