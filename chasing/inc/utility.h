#ifndef CPU_POINTER_CHASING_UTILITY_H_
#define CPU_POINTER_CHASING_UTILITY_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chase {

using index_t = std::uint32_t;

// 题目参数
constexpr index_t kChainDivisor = 64;  // 默认链数量 = 表大小 / 64
constexpr index_t kStepMin = 32;
constexpr index_t kStepMax = 256;
constexpr index_t kMinTableSize = 64;
constexpr index_t kMaxTableSize = 0x7FFFFFC0u;
constexpr int kBenchmarkRounds = 3;
constexpr const char* kDefaultTableFile = "my_table.bin";

// 跳转表文件格式
constexpr char kFileMagic[8] = {'C', 'P', 'C', 'H', 'A', 'S', 'E', '1'};
constexpr std::uint32_t kFileVersion = 1;
constexpr std::size_t kFileHeaderSize = 64;

struct Problem {
    index_t table_size;
    index_t chain_count;
    const index_t* table;
    const index_t* starts;
    const index_t* steps;
};

struct TableData {
    std::uint64_t seed = 0;
    index_t table_size = 0;
    index_t chain_count = 0;
    index_t step_min = 0;
    index_t step_max = 0;
    std::vector<index_t> table;
    std::vector<index_t> starts;
    std::vector<index_t> steps;

    Problem problem() const {
        return {table_size, chain_count, table.data(), starts.data(), steps.data()};
    }
};

// 选手实现，定义在 src/chasing.cpp。
void chase(const Problem& problem, index_t* final_pos);

// 题目辅助函数，定义在 src/utility.cpp。
bool generate_table(index_t table_size, index_t chain_count, std::uint64_t seed,
                    TableData& data, std::string& error);
bool write_table_file(const std::string& path, const TableData& data, std::string& error);
bool read_table_file(const std::string& path, TableData& data, std::string& error);
bool check_table_structure(const TableData& data, std::string& error);
void chase_reference(const Problem& problem, index_t* final_pos);
std::uint64_t scan_table(const TableData& data);

}  // namespace chase

#endif  // CPU_POINTER_CHASING_UTILITY_H_
