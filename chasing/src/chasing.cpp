#include "utility.h"

#include <algorithm>
#include <new>
#include <thread>
#include <vector>

namespace chase {
namespace {

constexpr index_t kInterleave = 16;
constexpr index_t kMaxThreads = 8;
#if 0  // Temporarily disable step-count bucketing.
// Initial cost guards: avoid sorting tiny jobs or sparsely populated buckets.
constexpr index_t kMinBucketChains = 4096;
constexpr index_t kMaxStepBuckets = 4096;
constexpr index_t kMinChainsPerBucket = 256;
constexpr index_t kMinAverageSteps = 16;
#endif

// Advance sixteen chains by their common remaining steps, then refill finished slots.
void chase_range(const Problem& problem, index_t* final_pos,
                 index_t begin, index_t end, const index_t* order = nullptr) {
    struct Slot {
        index_t chain;
        index_t pos;
        index_t remaining;
    };
    Slot slots[kInterleave];
    const index_t inactive = problem.chain_count;
    index_t next = begin;
    unsigned active = 0;
    for (auto& slot : slots) {
        slot = {inactive, 0, 0};
        if (next < end) {
            const index_t chain = order ? order[next - begin] : next;
            slot = {chain, problem.starts[chain], problem.steps[chain]};
            ++next;
            ++active;
        }
    }

    const index_t* table = problem.table;
    while (active == kInterleave) {
        const index_t common = std::min({
            slots[0].remaining, slots[1].remaining, slots[2].remaining, slots[3].remaining,
            slots[4].remaining, slots[5].remaining, slots[6].remaining, slots[7].remaining,
            slots[8].remaining, slots[9].remaining, slots[10].remaining, slots[11].remaining,
            slots[12].remaining, slots[13].remaining, slots[14].remaining, slots[15].remaining});
        // Advance every chain four times, then finish the remaining 0-3 steps.
        for (index_t blocks = common / 4; blocks != 0; --blocks) {
            slots[0].pos = table[slots[0].pos];
            slots[1].pos = table[slots[1].pos];
            slots[2].pos = table[slots[2].pos];
            slots[3].pos = table[slots[3].pos];
            slots[4].pos = table[slots[4].pos];
            slots[5].pos = table[slots[5].pos];
            slots[6].pos = table[slots[6].pos];
            slots[7].pos = table[slots[7].pos];
            slots[8].pos = table[slots[8].pos];
            slots[9].pos = table[slots[9].pos];
            slots[10].pos = table[slots[10].pos];
            slots[11].pos = table[slots[11].pos];
            slots[12].pos = table[slots[12].pos];
            slots[13].pos = table[slots[13].pos];
            slots[14].pos = table[slots[14].pos];
            slots[15].pos = table[slots[15].pos];

            slots[0].pos = table[slots[0].pos];
            slots[1].pos = table[slots[1].pos];
            slots[2].pos = table[slots[2].pos];
            slots[3].pos = table[slots[3].pos];
            slots[4].pos = table[slots[4].pos];
            slots[5].pos = table[slots[5].pos];
            slots[6].pos = table[slots[6].pos];
            slots[7].pos = table[slots[7].pos];
            slots[8].pos = table[slots[8].pos];
            slots[9].pos = table[slots[9].pos];
            slots[10].pos = table[slots[10].pos];
            slots[11].pos = table[slots[11].pos];
            slots[12].pos = table[slots[12].pos];
            slots[13].pos = table[slots[13].pos];
            slots[14].pos = table[slots[14].pos];
            slots[15].pos = table[slots[15].pos];

            slots[0].pos = table[slots[0].pos];
            slots[1].pos = table[slots[1].pos];
            slots[2].pos = table[slots[2].pos];
            slots[3].pos = table[slots[3].pos];
            slots[4].pos = table[slots[4].pos];
            slots[5].pos = table[slots[5].pos];
            slots[6].pos = table[slots[6].pos];
            slots[7].pos = table[slots[7].pos];
            slots[8].pos = table[slots[8].pos];
            slots[9].pos = table[slots[9].pos];
            slots[10].pos = table[slots[10].pos];
            slots[11].pos = table[slots[11].pos];
            slots[12].pos = table[slots[12].pos];
            slots[13].pos = table[slots[13].pos];
            slots[14].pos = table[slots[14].pos];
            slots[15].pos = table[slots[15].pos];

            slots[0].pos = table[slots[0].pos];
            slots[1].pos = table[slots[1].pos];
            slots[2].pos = table[slots[2].pos];
            slots[3].pos = table[slots[3].pos];
            slots[4].pos = table[slots[4].pos];
            slots[5].pos = table[slots[5].pos];
            slots[6].pos = table[slots[6].pos];
            slots[7].pos = table[slots[7].pos];
            slots[8].pos = table[slots[8].pos];
            slots[9].pos = table[slots[9].pos];
            slots[10].pos = table[slots[10].pos];
            slots[11].pos = table[slots[11].pos];
            slots[12].pos = table[slots[12].pos];
            slots[13].pos = table[slots[13].pos];
            slots[14].pos = table[slots[14].pos];
            slots[15].pos = table[slots[15].pos];
        }
        for (index_t tail = common % 4; tail != 0; --tail) {
            slots[0].pos = table[slots[0].pos];
            slots[1].pos = table[slots[1].pos];
            slots[2].pos = table[slots[2].pos];
            slots[3].pos = table[slots[3].pos];
            slots[4].pos = table[slots[4].pos];
            slots[5].pos = table[slots[5].pos];
            slots[6].pos = table[slots[6].pos];
            slots[7].pos = table[slots[7].pos];
            slots[8].pos = table[slots[8].pos];
            slots[9].pos = table[slots[9].pos];
            slots[10].pos = table[slots[10].pos];
            slots[11].pos = table[slots[11].pos];
            slots[12].pos = table[slots[12].pos];
            slots[13].pos = table[slots[13].pos];
            slots[14].pos = table[slots[14].pos];
            slots[15].pos = table[slots[15].pos];
        }

        for (auto& slot : slots) {
            slot.remaining -= common;
            if (slot.remaining != 0) continue;
            final_pos[slot.chain] = slot.pos;
            if (next < end) {
                const index_t chain = order ? order[next - begin] : next;
                slot = {chain, problem.starts[chain], problem.steps[chain]};
                ++next;
            } else {
                slot.chain = inactive;
                --active;
            }
        }
    }

    // Once input is exhausted, finish the remaining occupied slots.
    for (auto& slot : slots) {
        if (slot.chain == inactive) continue;
        for (; slot.remaining != 0; --slot.remaining)
            slot.pos = table[slot.pos];
        final_pos[slot.chain] = slot.pos;
    }
}


#if 0  // Retained for later comparison with direct scheduling.
// Bucket only chain IDs; the table, starts, steps and output indices stay intact.
void chase_bucketed_range(const Problem& problem, index_t* final_pos,
                          index_t begin, index_t end) {
    const index_t count = end - begin;
    if (count < kMinBucketChains) {
        chase_range(problem, final_pos, begin, end);
        return;
    }

    index_t min_steps = problem.steps[begin];
    index_t max_steps = min_steps;
    std::uint64_t total_steps = 0;
    for (index_t c = begin; c < end; ++c) {
        min_steps = std::min(min_steps, problem.steps[c]);
        max_steps = std::max(max_steps, problem.steps[c]);
        total_steps += problem.steps[c];
    }
    const std::uint64_t bucket_count =
        static_cast<std::uint64_t>(max_steps) - min_steps + 1;
    if (bucket_count == 1 || bucket_count > kMaxStepBuckets ||
        bucket_count > count / kMinChainsPerBucket ||
        total_steps < static_cast<std::uint64_t>(count) * kMinAverageSteps) {
        chase_range(problem, final_pos, begin, end);
        return;
    }

    // Counting sort costs O(chains + step range), with one index per chain.
    std::vector<index_t> offsets, cursor, order;
    try {
        offsets.resize(static_cast<std::size_t>(bucket_count) + 1, 0);
        cursor.resize(static_cast<std::size_t>(bucket_count));
        order.resize(count);
    } catch (const std::bad_alloc&) {
        chase_range(problem, final_pos, begin, end);
        return;
    }
    for (index_t c = begin; c < end; ++c)
        ++offsets[problem.steps[c] - min_steps + 1];
    for (std::size_t b = 0; b < bucket_count; ++b) {
        offsets[b + 1] += offsets[b];
        cursor[b] = offsets[b];
    }
    for (index_t c = begin; c < end; ++c)
        order[cursor[problem.steps[c] - min_steps]++] = c;

    // Keep equal-length buckets separate so full groups finish together.
    // Density guards limit the fraction of chains in per-bucket serial tails.
    for (std::size_t b = 0; b < bucket_count; ++b) {
        const index_t size = offsets[b + 1] - offsets[b];
        if (size != 0)
            chase_range(problem, final_pos, 0, size, order.data() + offsets[b]);
    }
}

#endif

}  // namespace

void chase(const Problem& problem, index_t* final_pos) {
    const index_t count = problem.chain_count;
    if (count == 0) return;
    const index_t threads = std::min(kMaxThreads,
        std::max<index_t>(1, count / kInterleave));
    if (threads == 1) {
        // chase_bucketed_range(problem, final_pos, 0, count);
        chase_range(problem, final_pos, 0, count);
        return;
    }

    // Split by chain count, without estimating each chain's work.
    const index_t per_thread = count / threads;
    const index_t extra = count % threads;
    auto run = [&](index_t tid) {
        const index_t begin = tid * per_thread + std::min(tid, extra);
        const index_t end = begin + per_thread + (tid < extra);
        // chase_bucketed_range(problem, final_pos, begin, end);
        chase_range(problem, final_pos, begin, end);
    };

    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    try {
        for (index_t tid = 1; tid < threads; ++tid)
            workers.emplace_back(run, tid);
    } catch (...) {
        // Join existing workers if another thread cannot be created.
        for (auto& worker : workers) worker.join();
        throw;
    }
    run(0);
    for (auto& worker : workers) worker.join();
}

}  // namespace chase
