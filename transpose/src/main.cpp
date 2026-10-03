#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <random>
#include <utility>
#include <cmath>
#include <stdexcept>
#include<assert.h>
#include <cuda_runtime.h>
#include "trans.h"

namespace{
static const unsigned int kSeedConstant = 0x9E3779B9u;
static const double maxTolerateError = 1.0e-3;

struct Case {
    int N;
};
struct CaseResult {
    Case test_case;
    double ref_trans_ms;
    double impl_trans_ms;
    double speedup_trans;
    double error_trnas;
    bool ok_trans;
    double score;
    double weight;
    std::string policy_name;
};
struct JudgePolicy{
    std::string name;
    int ref_speed_up;
    int accurate_score;
    int speed_up_score;
    int weight;
};
static const JudgePolicy nSmallPolicy = {"small", 200, 30, 70, 30};
static const JudgePolicy nMeduimPolicy = {"medium", 450, 30, 70, 30};
static const JudgePolicy nLargePolicy = {"large", 700, 30, 70, 30};
const int boundSmallMedium = 600;
const int boundMeduimLarge = 2000;
const double maxScore = 100;


void print_usage(const char *program) {
    std::cerr << "usage:\n";
    std::cerr << "  " << program << " CASE_FILE\n";
}
bool load_cases_from_file(const char *path, std::vector<Case> &cases) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "error: cannot open file '" << path << "'\n";
        return false;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;

        std::size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos == std::string::npos) {
            continue;
        }

        if (line[pos] == '#') {
            continue;
        }

        std::istringstream iss(line);
        int value;
        if (!(iss >> value)) {
            std::cerr << "error: line " << line_no
                      << " is not an integer: '" << line << "'\n";
            return false;
        }
        std::string extra;
        if (iss >> extra) {
            std::cerr << "error: line " << line_no
                      << " has unexpected trailing content: '" << line << "'\n";
            return false;
        }

        if (value % 32 != 0) {
            throw std::runtime_error("The " + std::to_string(cases.size()) + "-th case is not a multiple of 32!\n");
        }

        cases.push_back((Case){value});
    }

    if (cases.empty()) {
        std::cerr << "error: no case data found in file '" << path << "'\n";
        return false;
    }
    
    return true;
}

double run_trans(void (*function)(int, const double *, double*), int N, const double *A, double *B){
    //for warm-up
    function(N, A, B);

    cudaDeviceSynchronize();
    const auto start = std::chrono::high_resolution_clock::now();
    function(N, A, B);
    cudaDeviceSynchronize();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

JudgePolicy get_policy(int N) {
    if(N < boundSmallMedium) {
        return nSmallPolicy;
    } else if (N < boundMeduimLarge) {
        return nMeduimPolicy;
    } else {
        return nLargePolicy;
    }
}

void ref_MatrixTranspose(int N, const double *A, double *B){
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            B[j * N + i] = A[i * N + j];
        }
    }
    for (int i = 0; i < N; ++i) {
        for(int j = 0; j < N; j+=32) {
            for(int k = 0; k < 8; ++k) {
                double tmp = B[i * N + j];
                for(int l = 0; l < 32 ; ++l){
                    B[i * N + j + l] += (l == 31) ? tmp : B[i * N + j + l + 1];
                }
            }
        }
    }
}

CaseResult run_case(Case test){
    const int N = test.N;
    CaseResult result;
    result.test_case=test;

    const unsigned int seed = static_cast<unsigned int>(N) ^ kSeedConstant;
    std::mt19937 gen(seed);

    std::vector<double> A(N * N);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = dist(gen);
    }

    std::vector<double> B(A.size());
    std::vector<double> B_ref(A.size());

    double *d_A = nullptr;
    double *d_B = nullptr;
    cudaMalloc(&d_A, A.size() * sizeof(double));
    cudaMalloc(&d_B, A.size() * sizeof(double));
    cudaMemcpy(d_A, A.data(), A.size() * sizeof(double),cudaMemcpyHostToDevice);

    result.impl_trans_ms=run_trans(MatrixTranspose, N, d_A, d_B);
    result.ref_trans_ms=run_trans(ref_MatrixTranspose, N, A.data(), B_ref.data());

    cudaMemcpy(B.data(), d_B, A.size() * sizeof(double), cudaMemcpyDeviceToHost);

    double max_error = 0.0;
    for (std::size_t i = 0; i < B.size(); ++i) {
        const double diff = std::fabs(B[i] - B_ref[i]);
        if (diff > max_error) {
            max_error = diff;
        }
    }
    result.error_trnas = max_error;
    result.ok_trans = (result.error_trnas < maxTolerateError);

    result.speedup_trans = result.ref_trans_ms / result.impl_trans_ms;
    if (!result.ok_trans) {
        result.speedup_trans = 0;
    }

    cudaFree(d_A);
    cudaFree(d_B);

    JudgePolicy cur_policy = get_policy(N);
    result.policy_name = cur_policy.name;
    if(result.ok_trans) {
        result.score = cur_policy.accurate_score;
    } else {
        result.score = 0;
    }
    result.score += std::min(result.speedup_trans / cur_policy.ref_speed_up, 1.0) * cur_policy.speed_up_score;
    result.weight = cur_policy.weight;

    return result;
}

void normalize_scores(std::vector<CaseResult> &results) {
    double sum_weights = 0;
    for(auto result : results) {
        sum_weights += result.weight;
    }
    for(auto &result : results) {
        result.weight = result.weight / sum_weights * maxScore;
    }
    return;
}
void print_case_result(CaseResult resultm, std::size_t index, std::size_t total) {
    std::cout << "===== case " << index << "/" << total << " (N = " << resultm.test_case.N << ", size type=" << resultm.policy_name <<  ") =====\n";
    std::cout << "  reference time  : " << resultm.ref_trans_ms << " ms\n";
    std::cout << "  impl time       : " << resultm.impl_trans_ms << " ms\n";
    std::cout << "  speedup         : " << resultm.speedup_trans << " x\n";
    std::cout << "  error           : " << resultm.error_trnas << "\n";
    std::cout << "  status          : " << (resultm.ok_trans ? "OK" : "FAIL") << "\n";
    std::cout << "  score           : " << resultm.score << " / " << maxScore << "\n";
    std::cout << "  weight          : " << resultm.weight << "\n";
}
void print_summary(std::vector<CaseResult> results){
    std::cout << ">>=== summary ===<<\n";
    std::cout << "===== all " << results.size() <<" case(s)=====\n";
    double sum_score = 0;
    double sum_speed_up = 0;
    bool ok_flag = true;
    for(auto result : results) {
        sum_score += result.score * result.weight;
        sum_speed_up += result.speedup_trans * result.weight;
        if(!result.ok_trans) {
            ok_flag = false;
        }
    }
    sum_score /= maxScore;
    sum_speed_up /= maxScore;
    std::cout << "  status          : " << (ok_flag ? "OK" : "FAIL") << "\n";
    std::cout << "  average speed up: " << sum_speed_up << " x\n";
    std::cout << "  total score     : " << sum_score << "\n";
}
}//namespace
int main(int argc, char *argv[]) {
    std::cout << std::fixed << std::setprecision(3);

    std::vector<Case> cases;

    if (argc == 2) {
        if (!load_cases_from_file(argv[1], cases)) {
            return 1;
        }
    } else {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<CaseResult> results;
    results.reserve(cases.size());
    for (const auto test_case : cases) {
        results.push_back(run_case(test_case));
    }
    normalize_scores(results);
    for (std::size_t i = 0; i < results.size(); ++i) {
        print_case_result(results[i], i+1, results.size());
    }
    print_summary(results);
}