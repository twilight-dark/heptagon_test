#include <cstddef>
#include <vector>

std::vector<int> histogram_baseline(int N, int M, const std::vector<int>& in) {
  if (M <= 0) {
    return {};
  }

  std::vector<int> out(static_cast<std::size_t>(M), 0);
  for (int index = 0; index < N; ++index) {
    ++out[static_cast<std::size_t>(in[static_cast<std::size_t>(index)])];
  }
  return out;
}