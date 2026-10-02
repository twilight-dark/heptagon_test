#include "utility.h"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

char lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

// Parse a non-negative integer. 0x and K/M/G suffixes are supported.
bool parse_number(const std::string& text, std::uint64_t& result) {
    if (text.empty()) return false;

    std::size_t pos = 0;
    int base = 10;
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        base = 16;
        pos = 2;
    }

    std::uint64_t value = 0;
    std::size_t digits = 0;
    while (pos < text.size()) {
        const char c = text[pos];
        int digit = -1;
        if (c >= '0' && c <= '9') digit = c - '0';
        if (base == 16 && c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        if (base == 16 && c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        if (digit < 0 || digit >= base) break;

        if (value > (UINT64_MAX - static_cast<std::uint64_t>(digit)) /
                        static_cast<std::uint64_t>(base)) {
            return false;
        }
        value = value * static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(digit);
        ++pos;
        ++digits;
    }
    if (digits == 0) return false;

    if (pos < text.size()) {
        std::uint64_t multiplier = 0;
        switch (lower(text[pos])) {
            case 'k': multiplier = 1ull << 10; break;
            case 'm': multiplier = 1ull << 20; break;
            case 'g': multiplier = 1ull << 30; break;
            default: return false;
        }
        ++pos;
        if (pos < text.size() && lower(text[pos]) == 'i') ++pos;
        if (pos < text.size() && lower(text[pos]) == 'b') ++pos;
        if (pos != text.size() || value > UINT64_MAX / multiplier) return false;
        value *= multiplier;
    }

    result = value;
    return true;
}

void print_usage(const char* program) {
    std::cout
        << "CPU Pointer Chasing Challenge\n\n"
        << "Usage:\n"
        << "  " << program
        << " gen   -c <table-size> [-o <output-file>] [-s <seed>] [--chains <count>]\n"
        << "  " << program << " bench [-i <input-file>] [--no-baseline]\n\n"
        << "Examples:\n"
        << "  " << program << " gen -c 16M -s 1 -o my_table.bin\n"
        << "  " << program << " gen -c 16M -s 1 --chains 4096 -o c4096.bin\n"
        << "  " << program << " bench -i my_table.bin\n"
        << "  " << program << " bench -i my_table.bin --no-baseline\n";
}

}  // namespace

int main(int argc, char** argv) {
    using namespace chase;

    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    const std::string command = argv[1];

    // -----------------------------------------------------------------------
    // Generate a jump-table file.
    // -----------------------------------------------------------------------
    if (command == "gen") {
        std::uint64_t table_size = 0;
        std::uint64_t chain_count = 0;
        std::uint64_t seed = 0;
        bool has_table_size = false;
        bool has_seed = false;
        std::string output = kDefaultTableFile;

        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if ((option == "-c" || option == "-o" || option == "-s" ||
                 option == "--chains") && i + 1 >= argc) {
                std::cerr << "Error: option " << option << " requires a value.\n";
                return 1;
            }

            if (option == "-c") {
                if (!parse_number(argv[++i], table_size)) {
                    std::cerr << "Error: invalid table size.\n";
                    return 1;
                }
                has_table_size = true;
            } else if (option == "-o") {
                output = argv[++i];
            } else if (option == "-s") {
                if (!parse_number(argv[++i], seed)) {
                    std::cerr << "Error: invalid seed.\n";
                    return 1;
                }
                has_seed = true;
            } else if (option == "--chains") {
                if (!parse_number(argv[++i], chain_count) || chain_count == 0) {
                    std::cerr << "Error: invalid chain count.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: unknown option " << option << ".\n";
                return 1;
            }
        }

        if (!has_table_size || table_size < kMinTableSize || table_size > kMaxTableSize ||
            table_size % kChainDivisor != 0) {
            std::cerr << "Error: -c must be a multiple of 64 in the range [64, "
                      << kMaxTableSize << "].\n";
            return 1;
        }
        if (chain_count == 0) chain_count = table_size / kChainDivisor;
        if (chain_count > table_size) {
            std::cerr << "Error: chain count cannot exceed table size.\n";
            return 1;
        }

        if (!has_seed) {
            std::random_device random_device;
            seed = (static_cast<std::uint64_t>(random_device()) << 32) ^ random_device();
            seed ^= static_cast<std::uint64_t>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::cout << "No seed specified. Using seed " << seed << ".\n";
        }

        TableData data;
        std::string error;
        if (!generate_table(static_cast<index_t>(table_size),
                            static_cast<index_t>(chain_count), seed, data, error) ||
            !write_table_file(output, data, error)) {
            std::cerr << "Generation failed: " << error << '\n';
            return 1;
        }

        const std::uint64_t file_size = kFileHeaderSize +
            (static_cast<std::uint64_t>(data.table_size) + 2ull * data.chain_count) *
                sizeof(index_t);
        std::cout << std::fixed << std::setprecision(1)
                  << "Generation complete\n"
                  << "  Output file      : " << output << '\n'
                  << "  File size        : " << file_size / (1024.0 * 1024.0) << " MiB\n"
                  << "  Table size N     : " << data.table_size << '\n'
                  << "  Chain count C    : " << data.chain_count << '\n'
                  << "  Steps per chain  : " << data.step_min << " - " << data.step_max << '\n'
                  << "  Seed             : " << data.seed << '\n'
                  << "Reproduction command:\n  " << argv[0] << " gen -c " << data.table_size
                  << " -s " << data.seed << " --chains " << data.chain_count
                  << " -o " << output << '\n';
        return 0;
    }

    // -----------------------------------------------------------------------
    // Read, time, and optionally validate a jump table.
    // -----------------------------------------------------------------------
    if (command == "bench") {
        std::string input = kDefaultTableFile;
        bool run_baseline = true;

        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "-i") {
                if (i + 1 >= argc) {
                    std::cerr << "Error: option -i requires a file path.\n";
                    return 1;
                }
                input = argv[++i];
            } else if (option == "--no-baseline") {
                run_baseline = false;
            } else {
                std::cerr << "Error: unknown bench option " << option << ".\n";
                return 1;
            }
        }

        TableData data;
        std::string error;
        if (!read_table_file(input, data, error)) {
            std::cerr << "Read failed: " << error << '\n';
            return 1;
        }

        const Problem problem = data.problem();
        std::uint64_t total_jumps = 0;
        for (index_t steps : data.steps) total_jumps += steps;

        std::cout << std::fixed << std::setprecision(2)
                  << "Input file       : " << input << '\n'
                  << "Table size N     : " << data.table_size << " entries ("
                  << data.table_size * sizeof(index_t) / (1024.0 * 1024.0) << " MiB)\n"
                  << "Chain count C    : " << data.chain_count << '\n'
                  << "Steps per chain  : " << data.step_min << " - " << data.step_max << '\n'
                  << "Total jumps      : " << total_jumps << '\n'
                  << "Timing rounds    : " << kBenchmarkRounds << '\n'
                  << "Baseline/check   : " << (run_baseline ? "enabled" : "disabled") << '\n';

        const std::uint64_t scan_sum = scan_table(data);
        std::cout << "Warm-up checksum : 0x" << std::hex << scan_sum << std::dec << '\n';

        std::vector<index_t> result(data.chain_count);
        double reference_average = 0.0;
        std::vector<index_t> expected;

        if (run_baseline) {
            expected.resize(data.chain_count);
            double reference_total = 0.0;
            for (int round = 0; round < kBenchmarkRounds; ++round) {
                scan_table(data);
                std::atomic_signal_fence(std::memory_order_seq_cst);
                const auto begin = std::chrono::steady_clock::now();
                chase_reference(problem, expected.data());
                const auto end = std::chrono::steady_clock::now();
                std::atomic_signal_fence(std::memory_order_seq_cst);
                reference_total += std::chrono::duration<double>(end - begin).count();
            }
            reference_average = reference_total / kBenchmarkRounds;
        }

        double contestant_total = 0.0;
        for (int round = 0; round < kBenchmarkRounds; ++round) {
            scan_table(data);
            std::atomic_signal_fence(std::memory_order_seq_cst);
            const auto begin = std::chrono::steady_clock::now();
            chase::chase(problem, result.data());
            const auto end = std::chrono::steady_clock::now();
            std::atomic_signal_fence(std::memory_order_seq_cst);
            contestant_total += std::chrono::duration<double>(end - begin).count();
        }
        const double contestant_average = contestant_total / kBenchmarkRounds;

        if (run_baseline) {
            for (index_t c = 0; c < data.chain_count; ++c) {
                if (result[c] != expected[c]) {
                    std::cerr << "Validation failed at chain " << c << ": expected "
                              << expected[c] << ", got " << result[c] << ".\n";
                    return 2;
                }
            }
        }

        const double ns_per_jump = contestant_average * 1e9 / total_jumps;
        const double jumps_per_second = total_jumps / contestant_average;

        std::cout << std::fixed
                  << "\n=== Average results (" << kBenchmarkRounds << " rounds) ===\n"
                  << std::setprecision(6);
        if (run_baseline) {
            std::cout << "  L0 baseline time : " << reference_average << " s\n";
        }
        std::cout << "  Contestant time  : " << contestant_average << " s\n"
                  << std::setprecision(2)
                  << "  Time per jump    : " << ns_per_jump << " ns\n"
                  << std::setprecision(1)
                  << "  Jump throughput  : " << jumps_per_second / 1e6 << " M jumps/s\n";
        if (run_baseline) {
            std::cout << std::setprecision(2)
                      << "  Speedup over L0  : " << reference_average / contestant_average << " x\n"
                      << "  Validation       : passed\n";
        }

        const double workload_time = run_baseline ? reference_average : contestant_average;
        if (workload_time < 0.01) {
            std::cout << "\nWarning: the workload is small and timing may be noisy. "
                         "Increase the table size or chain count.\n";
        }
        return 0;
    }

    std::cerr << "Error: unknown command " << command << ".\n";
    print_usage(argv[0]);
    return 1;
}
