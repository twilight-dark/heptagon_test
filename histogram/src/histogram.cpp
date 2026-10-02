#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <omp.h>

namespace {
// Initial routing policy, not measured crossover points.
constexpr int kSerialMaxN = 32768;
constexpr std::size_t kPrivateBudget = 16 * 1024 * 1024;
constexpr std::size_t kCacheLineInts = 64 / sizeof(int);
// Initial cost heuristics, not machine-specific optimal crossover points.
constexpr std::size_t kSmallPrivateBytes = 256 * 1024;
constexpr std::size_t kSampleLimit = 1024;
constexpr std::size_t kSampleRegions = 256;

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

std::uint64_t sample_random(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

}  // namespace

std::vector<int> histogram_serial(int N, int M, const std::vector<int>& in) {
    if (M <= 0) return {};
    std::vector<int> out(static_cast<std::size_t>(M), 0);
    for (int i = 0; i < N; ++i) ++out[in[i]];
    return out;
}

std::vector<int> histogram_batch_atomic(int N, int M, const std::vector<int>& in) {
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

std::vector<int> histogram_direct_atomic(int N, int M, const std::vector<int>& in) {
    if (M <= 0) return {};
    std::vector<int> out(static_cast<std::size_t>(M), 0);
    if (N <= 0) return out;
    const int threads = std::min(N, omp_get_max_threads());
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < N; ++i) {
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

        // Each worker owns a contiguous output range. The first r workers
        // receive one extra bin; M < actual_threads permits empty ranges.
        const std::size_t bins = static_cast<std::size_t>(M);
        const std::size_t q = bins / actual_threads;
        const std::size_t r = bins % actual_threads;
        const std::size_t begin = tid * q + std::min<std::size_t>(tid, r);
        const std::size_t end = begin + q + (static_cast<std::size_t>(tid) < r);
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
    const std::size_t n = static_cast<std::size_t>(N);
    const std::size_t samples = std::clamp<std::size_t>(n / 4096, 64, kSampleLimit);
    std::array<int, kSampleLimit> values;
    std::array<std::size_t, kSampleRegions> regions{};
    // Reproducible local PRNG: no global RNG state or known-input fingerprints.
    // One random position per stratum covers the full prefix [0, N), without
    // duplicate positions. Sampling selects an algorithm, never an output.
    std::uint64_t state = 0x243f6a8885a308d3ULL;
    const std::size_t q = n / samples, r = n % samples;
    for (std::size_t s = 0; s < samples; ++s) {
        const std::size_t begin = s * q + std::min(s, r);
        const std::size_t length = q + (s < r);
        const int value = in[begin + sample_random(state) % length];
        values[s] = value;
        ++regions[static_cast<std::uint64_t>(value) * kSampleRegions / M];
    }
    std::sort(values.begin(), values.begin() + samples);
    std::size_t distinct = 0, longest = 0, run = 0;
    for (std::size_t s = 0; s < samples; ++s) {
        if (s == 0 || values[s] != values[s - 1]) {
            ++distinct;
            run = 0;
        }
        longest = std::max(longest, ++run);
    }

    const std::size_t stride = private_stride(M);
    const std::size_t private_limit = std::max<std::size_t>(
        1, kPrivateBudget / sizeof(int) / stride);
    const int private_threads = static_cast<int>(std::min<std::size_t>(
        std::min(N, omp_get_max_threads()), private_limit));
    const bool fits_budget = stride <= kPrivateBudget / sizeof(int);
    const std::size_t private_work = stride * static_cast<std::size_t>(private_threads);
    // Compare initialization/merge volume with input work, not sample identity.
    const bool affordable = fits_budget && n >= 2 * private_work;
    if (fits_budget && static_cast<std::size_t>(M) <= kSmallPrivateBytes / sizeof(int) &&
        n >= private_work)
        return histogram_private(N, M, in);
    if (longest * 8 >= samples * 7 && affordable)
        return histogram_private(N, M, in);
    if (distinct * 4 <= samples)
        return histogram_batch_atomic(N, M, in);
    const bool concentrated = distinct * 4 <= samples * 3 ||
        *std::max_element(regions.begin(), regions.end()) * 8 >= samples;
    if (concentrated && affordable) return histogram_private(N, M, in);
    return histogram_direct_atomic(N, M, in);
}
