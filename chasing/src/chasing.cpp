#include "utility.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace chase {
namespace {

constexpr index_t kInterleave = 16;
constexpr index_t kMaxThreads = 8;

// Advance sixteen chains by their common remaining steps, then refill finished slots.
void chase_range(const Problem& problem, index_t* final_pos,
                 index_t begin, index_t end) {
    struct Slot {
        index_t chain;
        index_t pos;
        index_t remaining;
    };
    Slot slots[kInterleave];
    index_t next = begin;
    unsigned active = 0;
    for (auto& slot : slots) {
        slot = {end, 0, 0};
        if (next < end) {
            slot = {next, problem.starts[next], problem.steps[next]};
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
        for (index_t step = 0; step < common; ++step) {
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
                slot = {next, problem.starts[next], problem.steps[next]};
                ++next;
            } else {
                slot.chain = end;
                --active;
            }
        }
    }

    // Once input is exhausted, finish the remaining occupied slots.
    for (auto& slot : slots) {
        if (slot.chain == end) continue;
        for (; slot.remaining != 0; --slot.remaining)
            slot.pos = table[slot.pos];
        final_pos[slot.chain] = slot.pos;
    }
}

}  // namespace

void chase(const Problem& problem, index_t* final_pos) {
    const index_t count = problem.chain_count;
    if (count == 0) return;
    const index_t threads = std::min(kMaxThreads,
        std::max<index_t>(1, count / kInterleave));
    if (threads == 1) {
        chase_range(problem, final_pos, 0, count);
        return;
    }

    // Split by chain count, without estimating each chain's work.
    const index_t per_thread = count / threads;
    const index_t extra = count % threads;
    auto run = [&](index_t tid) {
        const index_t begin = tid * per_thread + std::min(tid, extra);
        const index_t end = begin + per_thread + (tid < extra);
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
