#include "histogram.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <omp.h>

namespace {
// Initial routing policy, not measured crossover points.
constexpr int kSerialMaxN = 32768;
constexpr int kPrivateMaxM = 32768;
constexpr std::size_t kPrivateBudget = 64 * 1024 * 1024;
constexpr std::size_t kCacheLineInts = 64 / sizeof(int);
constexpr int kMergeBlock = 4096;

std::size_t private_stride(int M) {
    // A full padding line separates rows even if the allocation is unaligned.
    return static_cast<std::size_t>(M) + kCacheLineInts;
}
}  // namespace

std::vector<int> histogram_serial(int N, int M, const std::vector<int>& in) {
    if (M <= 0) return {};
    std::vector<int> out(static_cast<std::size_t>(M), 0);
    for (int i = 0; i < N; ++i) ++out[in[i]];
    return out;
}

std::vector<int> histogram_atomic(int N, int M, const std::vector<int>& in) {
    if (M <= 0) return {};
    std::vector<int> out(static_cast<std::size_t>(M), 0);
    if (N <= 0) return out;
    const int threads = std::min(N, omp_get_max_threads());
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < N; ++i) {
        // Only indivisible increments are needed; the implicit barrier makes
        // all updates complete before the result is returned.
        #pragma omp atomic update relaxed
        ++out[in[i]];
    }
    return out;
}

std::vector<int> histogram_private(int N, int M, const std::vector<int>& in) {
    if (M <= 0) return {};
    if (N <= 0) return std::vector<int>(static_cast<std::size_t>(M), 0);
    const std::size_t stride = private_stride(M);
    // Bound total private storage. At least one row is allowed for direct calls.
    const std::size_t budget_threads = std::max<std::size_t>(
        1, kPrivateBudget / sizeof(int) / stride);
    const int threads = static_cast<int>(std::min<std::size_t>(
        std::min(N, omp_get_max_threads()), budget_threads));
    // Default initialization leaves ints uninitialized; each worker zeros its row.
    std::unique_ptr<int[]> local(new int[stride * threads]);
    std::vector<int> out(static_cast<std::size_t>(M), 0);

    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int actual_threads = omp_get_num_threads();
        int* row = local.get() + static_cast<std::size_t>(tid) * stride;
        std::fill_n(row, M, 0);

        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) ++row[in[i]];
        // The preceding implicit barrier is required before reading other rows.

        #pragma omp for schedule(static)
        for (std::size_t begin = 0; begin < static_cast<std::size_t>(M);
             begin += kMergeBlock) {
            const std::size_t end = std::min(
                begin + kMergeBlock, static_cast<std::size_t>(M));
            for (int t = 0; t < actual_threads; ++t) {
                const int* source = local.get() + static_cast<std::size_t>(t) * stride;
                for (std::size_t bin = begin; bin < end; ++bin)
                    out[bin] += source[bin];
            }
        }
    }
    return out;
}

std::vector<int> histogram(int N, int M, const std::vector<int>& in) {
    if (M <= 0 || N <= kSerialMaxN || omp_get_max_threads() == 1)
        return histogram_serial(N, M, in);
    if (M <= kPrivateMaxM) return histogram_private(N, M, in);
    return histogram_atomic(N, M, in);
}
