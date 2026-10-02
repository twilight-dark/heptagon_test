#pragma once

#include <vector>

// For M > 0: N >= 0, in.size() >= N, and 0 <= in[i] < M for i < N.
// M <= 0 returns an empty vector. Only the first N elements are counted.
std::vector<int> histogram(int N, int M, const std::vector<int>& in);
std::vector<int> histogram_serial(int N, int M, const std::vector<int>& in);
std::vector<int> histogram_private(int N, int M, const std::vector<int>& in);
std::vector<int> histogram_atomic(int N, int M, const std::vector<int>& in);
