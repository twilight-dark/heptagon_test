#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// 选手实现
std::vector<int> histogram(int N, int M, const std::vector<int>& in);
// 发布的基线
std::vector<int> histogram_baseline(int N, int M,
                                    const std::vector<int>& in);

namespace {

// ---------- 可复现随机数 ----------
// splitmix64：无外部依赖，跨平台、跨标准库一致
uint64_t splitmix64(uint64_t& s) {
  s += 0x9E3779B97F4A7C15ULL;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// ---------- 分布 ----------
enum class Dist { Uniform, Hotspot, Single, Sequential };

void generate(std::vector<int>& in, int N, int M, Dist d, uint64_t seed) {
  in.resize(static_cast<std::size_t>(N));
  uint64_t s = seed;
  switch (d) {
    case Dist::Uniform:
      for (int i = 0; i < N; ++i)
        in[i] = static_cast<int>(splitmix64(s) % static_cast<uint64_t>(M));
      break;
    case Dist::Hotspot: {
      const uint64_t hot = static_cast<uint64_t>(std::max(1, M / 100));
      for (int i = 0; i < N; ++i) {
        const uint64_t r = splitmix64(s);
        in[i] = (r % 10 < 9)
                    ? static_cast<int>((r / 10) % hot)
                    : static_cast<int>((r / 10) % static_cast<uint64_t>(M));
      }
      break;
    }
    case Dist::Single:
      std::fill(in.begin(), in.end(), 0);
      break;
    case Dist::Sequential:
      for (int i = 0; i < N; ++i) in[i] = i % M;
      break;
  }
}

// ---------- Case 表 ----------
// 规模围绕 Xeon Gold 6338 的 cache 层级设计：
//   L1d 48KB / core, L2 1.25MB / core, L3 48MB / socket
// 关键：让 M 和 N 分别落在不同层级边界
struct Case {
  const char* name;
  int N;
  int M;
  Dist dist;
  uint64_t seed;
};

const Case kCases[] = {
    // 极小，全部落在 L1
    {"tiny_L1",            1000,       4,       Dist::Uniform,  1},

    // N 落 L2，M 小
    {"L2N_L1M",            100000,     1000,    Dist::Uniform,  2},

    // N=1e6，M 从最小到中等
    {"L2N_tinyM",          1000000,    4,       Dist::Uniform,  3},
    {"L2N_smallM_hot",     1000000,    1000,    Dist::Hotspot,  4},
    {"L2N_midM",           1000000,    20000,   Dist::Uniform,  5},

    // N=1e7，输入落 L3，M 跨层级
    {"L3N_tinyM",          10000000,   4,       Dist::Uniform,  6},
    {"L3N_smallM_hot",     10000000,   1000,    Dist::Hotspot,  7},
    {"L3N_largeM",         10000000,   1000000, Dist::Uniform,  8},
    {"L3N_largeM_hot",     10000000,   1000000, Dist::Hotspot,  9},

    // N=1e8，输入落 DRAM
    {"DRAMN_smallM",       100000000,  1000,    Dist::Uniform, 10},
    {"DRAMN_largeM",       100000000,  1000000, Dist::Uniform, 11},
    {"DRAMN_largeM_single",100000000,  1000000, Dist::Single,  12},
};

// ---------- 测速 ----------
// 单次计时
template <typename Fn>
double time_once(Fn fn, int N, int M, const std::vector<int>& in,
                 std::vector<int>& out) {
  const auto t0 = std::chrono::steady_clock::now();
  out = fn(N, M, in);
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int reps_for(int N) {
  if (N <= 100000)    return 10;
  if (N <= 10000000)  return 5;
  return 3;
}

// 列宽
constexpr int kColName    = 22;
constexpr int kColN       = 11;
constexpr int kColM       = 9;
constexpr int kColTime    = 11;
constexpr int kColSpeedup = 9;

void print_header() {
  std::cout << std::left  << std::setw(kColName) << "case"
            << std::right << std::setw(kColN)    << "N"
                          << std::setw(kColM)    << "M"
                          << std::setw(kColTime) << "base(ms)"
                          << std::setw(kColTime) << "sub(ms)"
                          << std::setw(kColSpeedup) << "speedup"
            << "\n";
  std::cout << std::string(kColName + kColN + kColM +
                           kColTime + kColTime + kColSpeedup, '-')
            << "\n";
}

}  // namespace

int main() {
  double total_base = 0.0;
  double total_sub  = 0.0;
  bool all_ok = true;

  print_header();

  for (const Case& c : kCases) {
    // base 和 sub 使用两份独立副本，物理地址不同，cache 状态对称
    std::vector<int> in_a, in_b;
    generate(in_a, c.N, c.M, c.dist, c.seed);
    generate(in_b, c.N, c.M, c.dist, c.seed);

    const int warmup = 1;
    const int reps   = reps_for(c.N);

    std::vector<int> ref, got;
    double base_ms = std::numeric_limits<double>::infinity();
    double sub_ms  = std::numeric_limits<double>::infinity();

    // 交替跑：每一轮先 base 后 sub，共享同频率、同负载、同 cache 状态
    for (int r = 0; r < warmup + reps; ++r) {
      const double b = time_once(histogram_baseline, c.N, c.M, in_a, ref);
      const double s = time_once(histogram,          c.N, c.M, in_b, got);
      if (r >= warmup) {
        base_ms = std::min(base_ms, b);
        sub_ms  = std::min(sub_ms,  s);
      }
    }

    const bool ok = (ref == got);
    if (!ok) all_ok = false;

    total_base += base_ms;
    total_sub  += sub_ms;

    std::cout << std::left  << std::setw(kColName) << c.name
              << std::right << std::setw(kColN) << c.N
                            << std::setw(kColM) << c.M
              << std::fixed << std::setprecision(3)
                            << std::setw(kColTime) << base_ms
                            << std::setw(kColTime) << sub_ms
              << std::fixed << std::setprecision(3)
                            << std::setw(kColSpeedup) << (base_ms / sub_ms)
              << (ok ? "" : "  [WRONG]")
              << "\n";
  }

  std::cout << std::string(kColName + kColN + kColM +
                           kColTime + kColTime + kColSpeedup, '-')
            << "\n";
  std::cout << "total: base=" << std::fixed << std::setprecision(3)
            << total_base << " ms, sub=" << total_sub
            << " ms, speedup=" << (total_base / total_sub) << "\n";
  return all_ok ? 0 : 1;
}