#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>
#include <omp.h>

namespace {
constexpr int kSerialMaxN = 32768;
constexpr std::size_t kPrivateBudget = 16 * 1024 * 1024;
constexpr std::size_t kCacheLineInts = 64 / sizeof(int);
}  // namespace

std::vector<int> histogram_serial(int N, int M, const std::vector<int>& in) {
    if (M <= 0) return {};
    std::vector<int> out(static_cast<std::size_t>(M), 0);
    for (int i = 0; i < N; ++i) ++out[in[i]];
    return out;
}

std::vector<int> histogram_private(int N, int M, const std::vector<int>& in) {
    if (M <= 0) return {};
    if (N <= 0) return std::vector<int>(static_cast<std::size_t>(M), 0);
    // Separate rows by a full cache line, even with an unaligned allocation.
    const std::size_t stride = static_cast<std::size_t>(M) + kCacheLineInts;
    // Bound total private storage. Always allow at least one row.
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

        // Accumulate each run locally, then update its bin once.
        int hot = -1;
        int hot_count = 0;
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < N; ++i) {
            const int value = in[i];
            if (value == hot) {
                ++hot_count;
            } else {
                if (hot_count != 0) row[hot] += hot_count;
                hot = value;
                hot_count = 1;
            }
        }
        if (hot_count != 0) row[hot] += hot_count;
        // Flush each worker's final run before any worker reads other rows.
        #pragma omp barrier

        // Merge disjoint output ranges in parallel; no atomic updates needed.
        const std::size_t bins = static_cast<std::size_t>(M);
        const std::size_t bins_per_thread = bins / actual_threads;
        const std::size_t extra_bins = bins % actual_threads;
        const std::size_t begin = tid * bins_per_thread +
                                  std::min<std::size_t>(tid, extra_bins);
        const std::size_t end = begin + bins_per_thread +
                                (static_cast<std::size_t>(tid) < extra_bins);
        for (int t = 0; t < actual_threads; ++t) {
            const int* source = local.get() + static_cast<std::size_t>(t) * stride;
            for (std::size_t bin = begin; bin < end; ++bin)
                out[bin] += source[bin];
        }
        // The parallel-region barrier completes all output ranges.
    }
    return out;
}

std::vector<int> histogram(int N, int M, const std::vector<int>& in) {
    if (M <= 0 || N <= kSerialMaxN || omp_get_max_threads() == 1)
        return histogram_serial(N, M, in);
    return histogram_private(N, M, in);
}
