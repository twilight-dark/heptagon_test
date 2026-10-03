#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "baseline.h"
#include "reduction.h"

namespace {
constexpr unsigned kSeedConstant = 0x9E3779B9u;
constexpr float kRelativeTolerance = 1.0e-4f;
constexpr std::size_t kCorrectnessLimit = std::size_t{1} << 20;
constexpr double kTargetSpeedup = 1024.0;
constexpr int kTimingSamples = 7;

struct CaseResult {
    std::size_t n = 0;
    float baseline_ms = 0.0f;
    float impl_ms = 0.0f;
    float relative_error = 0.0f;
    bool checked = false;
    bool ok = false;
    double score = 0.0;
};

void check(cudaError_t status, const char *what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

bool load_cases(const char *path, std::vector<std::size_t> &cases) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream parser(line);
        std::size_t n;
        std::string extra;
        if (!(parser >> n) || (parser >> extra) || n == 0) return false;
        cases.push_back(n);
    }
    return !cases.empty();
}

float reference(const std::vector<float> &input) {
    double sum = 0.0;
    for (float value : input) sum += static_cast<double>(value) * value;
    return static_cast<float>(sum);
}

template <class Function>
float elapsed_ms(Function &&fn) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");
    fn();
    check(cudaDeviceSynchronize(), "warm-up");
    std::array<float, kTimingSamples> samples{};
    for (float &sample : samples) {
        check(cudaEventRecord(start), "record start");
        fn();
        check(cudaEventRecord(stop), "record stop");
        check(cudaEventSynchronize(stop), "wait for kernel");
        check(cudaEventElapsedTime(&sample, start, stop), "elapsed time");
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(samples.begin(), samples.end());
    return samples[kTimingSamples / 2];
}

double score(bool correct, float baseline_ms, float impl_ms) {
    if (!correct) return 0.0;
    const double speedup = std::max(static_cast<double>(baseline_ms) / impl_ms, 1.0);
    const double progress = std::clamp(std::log2(speedup) / std::log2(kTargetSpeedup), 0.0, 1.0);
    return 30.0 + 70.0 * progress * progress;
}

CaseResult run_case(std::size_t n) {
    std::mt19937 gen(static_cast<unsigned>(n) ^ kSeedConstant);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input(n);
    for (float &x : input) x = dist(gen);
    const bool check_result = n < kCorrectnessLimit;
    const float expected = check_result ? reference(input) : 0.0f;

    float *d_input = nullptr, *d_baseline = nullptr, *d_result = nullptr, *d_workspace = nullptr;
    check(cudaMalloc(&d_input, n * sizeof(float)), "allocate input");
    check(cudaMalloc(&d_baseline, sizeof(float)), "allocate baseline result");
    check(cudaMalloc(&d_result, sizeof(float)), "allocate result");
    check(cudaMalloc(&d_workspace, n * sizeof(float)), "allocate workspace");
    check(cudaMemcpy(d_input, input.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy input");

    CaseResult result;
    result.n = n;
    result.baseline_ms = elapsed_ms([&] { BaselineReduction(n, d_input, d_baseline, d_workspace); });
    result.impl_ms = elapsed_ms([&] { HierarchicalReduction(n, d_input, d_result, d_workspace); });
    check(cudaGetLastError(), "kernel launch");
    result.checked = check_result;
    result.ok = true;
    if (check_result) {
        float actual;
        check(cudaMemcpy(&actual, d_result, sizeof(float), cudaMemcpyDeviceToHost), "copy result");
        result.relative_error = std::fabs(actual - expected) / std::max(std::fabs(expected), 1.0f);
        result.ok = result.relative_error <= kRelativeTolerance;
    }
    result.score = score(result.ok, result.baseline_ms, result.impl_ms);
    cudaFree(d_workspace);
    cudaFree(d_result);
    cudaFree(d_baseline);
    cudaFree(d_input);
    return result;
}

void print_case(const CaseResult &result) {
    std::cout << "===== N = " << result.n << " =====\n"
              << "  baseline time  : " << result.baseline_ms << " ms\n"
              << "  impl time      : " << result.impl_ms << " ms\n"
              << "  speedup        : " << result.baseline_ms / result.impl_ms << " x\n";
    if (result.checked) {
        std::cout << "  relative error : " << result.relative_error << "\n"
                  << "  status         : " << (result.ok ? "OK" : "FAIL") << "\n";
    } else {
        std::cout << "  relative error : N/A (timing only)\n"
                  << "  status         : TIME ONLY\n";
    }
    std::cout << "  score          : " << result.score << " / 100\n";
}
}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " CASE_FILE\n";
        return 1;
    }
    try {
        std::vector<std::size_t> cases;
        if (!load_cases(argv[1], cases)) {
            std::cerr << "error: invalid or empty case file '" << argv[1] << "'\n";
            return 1;
        }
        std::cout << std::fixed << std::setprecision(6);
        long double weighted_score = 0.0, total_weight = 0.0;
        bool correctness_ok = true;
        for (std::size_t n : cases) {
            const CaseResult result = run_case(n);
            print_case(result);
            weighted_score += static_cast<long double>(n) * result.score;
            total_weight += n;
            correctness_ok = correctness_ok && result.ok;
        }
        std::cout << ">>=== summary ===<<\n"
                  << "  total score    : "
                  << (correctness_ok ? static_cast<double>(weighted_score / total_weight) : 0.0)
                  << " / 100\n";
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
