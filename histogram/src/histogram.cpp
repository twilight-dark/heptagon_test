#include "histogram.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <omp.h>

namespace {
// Initial routing policy, not measured crossover points.
constexpr int kSerialMaxN = 32768;
constexpr int kPrivateMaxM = 32768;
constexpr std::size_t kPrivateBudget = 64 * 1024 * 1024;
constexpr std::size_t kCacheLineInts = 64 / sizeof(int);
constexpr int kMergeBlock = 4096;

// Keep the linear-probing table at most half full to bound typical lookup cost.
constexpr std::size_t kAtomicSlots = 1024;
constexpr std::size_t kAtomicCapacity = kAtomicSlots / 2;
static_assert((kAtomicSlots & (kAtomicSlots - 1)) == 0);

class AtomicBatch {
public:
    void add(int value, int* out) {
        std::size_t slot = hash(value);
        while (entries_[slot].count != 0) {
            if (entries_[slot].value == value) {
                ++entries_[slot].count;
                return;
            }
            slot = (slot + 1) & (kAtomicSlots - 1);
        }
        // Hits in a full batch remain local. Flush only for a new distinct value.
        if (used_ == kAtomicCapacity) {
            flush(out);
            slot = hash(value);
        }
        entries_[slot] = {value, 1};
        touched_[used_++] = slot;
    }

    void flush(int* out) {
        for (std::size_t i = 0; i < used_; ++i) {
            Entry& entry = entries_[touched_[i]];
            const int value = entry.value;
            const int count = entry.count;
            #pragma omp atomic update relaxed
            out[value] += count;
            entry.count = 0;
        }
        used_ = 0;
    }

private:
    static std::size_t hash(int value) {
        // Mix high bits too, so strided input does not all map to one slot.
        std::uint32_t mixed = static_cast<std::uint32_t>(value);
        mixed ^= mixed >> 16;
        mixed *= 0x7feb352dU;
        mixed ^= mixed >> 15;
        return mixed & (kAtomicSlots - 1);
    }

    struct Entry {
        int value;
        int count;
    };
    std::array<Entry, kAtomicSlots> entries_{};
    std::array<std::size_t, kAtomicCapacity> touched_;
    std::size_t used_ = 0;
};

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
    #pragma omp parallel num_threads(threads)
    {
        AtomicBatch batch;
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < N; ++i) batch.add(in[i], out.data());
        batch.flush(out.data());
        // All shared updates are atomic, including final partial batches.
        // The parallel-region barrier completes them before returning.
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
