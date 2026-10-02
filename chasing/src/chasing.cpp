#include "utility.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace chase {
namespace {

// Advance four independent chains together to overlap their memory accesses.
void chase_range(const Problem& problem, index_t* final_pos,
                 index_t begin, index_t end) {
    const index_t* table = problem.table;
    index_t c = begin;
    for (; end - c >= 4; c += 4) {
        index_t p0 = problem.starts[c];
        index_t p1 = problem.starts[c + 1];
        index_t p2 = problem.starts[c + 2];
        index_t p3 = problem.starts[c + 3];
        index_t s0 = problem.steps[c];
        index_t s1 = problem.steps[c + 1];
        index_t s2 = problem.steps[c + 2];
        index_t s3 = problem.steps[c + 3];

        while (s0 || s1 || s2 || s3) {
            if (s0) { p0 = table[p0]; --s0; }
            if (s1) { p1 = table[p1]; --s1; }
            if (s2) { p2 = table[p2]; --s2; }
            if (s3) { p3 = table[p3]; --s3; }
        }
        final_pos[c] = p0;
        final_pos[c + 1] = p1;
        final_pos[c + 2] = p2;
        final_pos[c + 3] = p3;
    }

    // Handle the remaining one to three chains.
    for (; c < end; ++c) {
        index_t pos = problem.starts[c];
        for (index_t remaining = problem.steps[c]; remaining != 0; --remaining)
            pos = table[pos];
        final_pos[c] = pos;
    }
}

}  // namespace

void chase(const Problem& problem, index_t* final_pos) {
    const index_t count = problem.chain_count;
    if (count == 0) return;
    const unsigned available = std::max(1u, std::thread::hardware_concurrency());
    const index_t threads = std::min<index_t>(available, std::max<index_t>(1, count / 4));
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
