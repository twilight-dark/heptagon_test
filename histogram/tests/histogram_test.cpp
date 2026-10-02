#include "../src/histogram.h"
#include "../baseline.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>

int main() {
    using Fn = std::vector<int> (*)(int, int, const std::vector<int>&);
    const Fn implementations[] = {
        histogram_serial, histogram_private, histogram_atomic, histogram};
    const char* names[] = {"serial", "private", "atomic", "dispatch"};
    std::mt19937 random(42);
    std::size_t checks = 0;
    for (int dynamic : {0, 1}) {
        omp_set_dynamic(dynamic);
        for (int threads : {1, 2, 3, 7, 8, 16}) {
            omp_set_num_threads(threads);
            for (int M : {-1, 0, 1, 4, 17, 1000, 4097, 32768, 32769, 1000000}) {
                for (int N : {0, 1, threads - 1, threads, threads + 1,
                              2 * threads + 1, 1003, 32768, 32769}) {
                    for (int distribution = 0; distribution < 5; ++distribution) {
                        // Invalid trailing values detect accidental processing beyond N.
                        std::vector<int> in(static_cast<std::size_t>(N) + 3, -1);
                        for (int i = 0; i < N && M > 0; ++i) {
                            if (distribution == 0) in[i] = random() % M;
                            if (distribution == 1) in[i] = M - 1;
                            if (distribution == 2) in[i] = i % M;
                            if (distribution == 3)
                                in[i] = (random() % 10 < 9) ? 0 : random() % M;
                            if (distribution == 4)
                                in[i] = (i == N - 1) ? 0 : M - 1;
                        }
                        const auto expected = histogram_baseline(N, M, in);
                        for (int f = 0; f < 4; ++f) {
                            const auto got = implementations[f](N, M, in);
                            if (got != expected || (M > 0 &&
                                std::accumulate(got.begin(), got.end(), std::int64_t{0}) != N)) {
                                std::cerr << names[f] << " failed: N=" << N
                                          << " M=" << M << " threads=" << threads
                                          << " dynamic=" << dynamic
                                          << " distribution=" << distribution << '\n';
                                return 1;
                            }
                            ++checks;
                        }
                    }
                }
            }
        }
    }
    std::cout << "Passed " << checks << " comparisons across all three implementations and dispatch.\n";
}
