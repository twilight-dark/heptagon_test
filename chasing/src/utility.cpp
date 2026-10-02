#if defined(__linux__) && !defined(_DEFAULT_SOURCE) && !defined(_GNU_SOURCE)
#define _DEFAULT_SOURCE 1
#endif

#include "utility.h"

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

using chase::index_t;

// SplitMix64 只使用固定宽度无符号整数，因此生成结果不依赖标准库实现。
struct SplitMix64 {
    std::uint64_t state;

    explicit SplitMix64(std::uint64_t seed) : state(seed) {}

    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    std::uint64_t bounded(std::uint64_t bound) {
        const std::uint64_t threshold = (0ull - bound) % bound;
        for (;;) {
            const std::uint64_t value = next();
            if (value >= threshold) return value % bound;
        }
    }
};

void put_u32le(std::uint8_t* p, std::uint32_t value) {
    p[0] = static_cast<std::uint8_t>(value);
    p[1] = static_cast<std::uint8_t>(value >> 8);
    p[2] = static_cast<std::uint8_t>(value >> 16);
    p[3] = static_cast<std::uint8_t>(value >> 24);
}

void put_u64le(std::uint8_t* p, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(value >> (8 * i));
}

std::uint32_t get_u32le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t get_u64le(const std::uint8_t* p) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return value;
}

bool little_endian() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

bool write_indices(std::FILE* file, const index_t* values, std::size_t count) {
    if (little_endian()) return std::fwrite(values, sizeof(index_t), count, file) == count;

    std::uint8_t bytes[4];
    for (std::size_t i = 0; i < count; ++i) {
        put_u32le(bytes, values[i]);
        if (std::fwrite(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) return false;
    }
    return true;
}

bool read_indices(std::FILE* file, index_t* values, std::size_t count) {
    if (little_endian()) return std::fread(values, sizeof(index_t), count, file) == count;

    std::uint8_t bytes[4];
    for (std::size_t i = 0; i < count; ++i) {
        if (std::fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) return false;
        values[i] = get_u32le(bytes);
    }
    return true;
}

std::uint64_t fnv1a(const void* data, std::size_t size,
                    std::uint64_t hash = 14695981039346656037ull) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t hash_indices(const index_t* values, std::size_t count, std::uint64_t hash) {
    if (little_endian()) return fnv1a(values, count * sizeof(index_t), hash);

    std::uint8_t bytes[4];
    for (std::size_t i = 0; i < count; ++i) {
        put_u32le(bytes, values[i]);
        hash = fnv1a(bytes, sizeof(bytes), hash);
    }
    return hash;
}

std::string number(std::uint64_t value) {
    return std::to_string(value);
}

}  // namespace

namespace chase {

bool generate_table(index_t table_size, index_t chain_count, std::uint64_t seed,
                    TableData& data, std::string& error) {
    if (table_size < kMinTableSize || table_size > kMaxTableSize ||
        table_size % kChainDivisor != 0) {
        error = "Table size must be a multiple of 64 in the range [64, " +
                number(kMaxTableSize) + "].";
        return false;
    }

    if (chain_count == 0) chain_count = table_size / kChainDivisor;
    if (chain_count > table_size) {
        error = "Chain count cannot exceed table size.";
        return false;
    }

    data.seed = seed;
    data.table_size = table_size;
    data.chain_count = chain_count;
    data.step_min = kStepMin;
    data.step_max = kStepMax;

    try {
        data.table.resize(table_size);
        data.starts.resize(chain_count);
        data.steps.resize(chain_count);
    } catch (const std::bad_alloc&) {
        error = "Insufficient memory.";
        return false;
    }

    SplitMix64 random(seed);
    for (index_t i = 0; i < table_size; ++i) data.table[i] = i;
    for (index_t i = table_size - 1; i > 0; --i) {
        const index_t j = static_cast<index_t>(random.bounded(static_cast<std::uint64_t>(i) + 1));
        const index_t temporary = data.table[i];
        data.table[i] = data.table[j];
        data.table[j] = temporary;
    }

    for (index_t i = 0; i < chain_count; ++i) data.starts[i] = data.table[i];

    const std::uint64_t range = static_cast<std::uint64_t>(kStepMax) - kStepMin + 1;
    for (index_t i = 0; i < chain_count; ++i) {
        data.steps[i] = static_cast<index_t>(kStepMin + random.bounded(range));
    }
    return true;
}

bool write_table_file(const std::string& path, const TableData& data, std::string& error) {
    std::uint8_t header[kFileHeaderSize] = {};
    std::memcpy(header, kFileMagic, sizeof(kFileMagic));
    put_u32le(header + 0x08, kFileVersion);
    put_u32le(header + 0x0C, sizeof(index_t));
    put_u64le(header + 0x10, data.table_size);
    put_u64le(header + 0x18, data.chain_count);
    put_u64le(header + 0x20, data.seed);
    put_u32le(header + 0x28, data.step_min);
    put_u32le(header + 0x2C, data.step_max);

    std::uint64_t checksum = fnv1a(header, 0x38);
    checksum = hash_indices(data.table.data(), data.table.size(), checksum);
    checksum = hash_indices(data.starts.data(), data.starts.size(), checksum);
    checksum = hash_indices(data.steps.data(), data.steps.size(), checksum);
    put_u64le(header + 0x38, checksum);

    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        error = "Cannot create file: " + path;
        return false;
    }

    bool ok = std::fwrite(header, 1, sizeof(header), file) == sizeof(header);
    ok = ok && write_indices(file, data.table.data(), data.table.size());
    ok = ok && write_indices(file, data.starts.data(), data.starts.size());
    ok = ok && write_indices(file, data.steps.data(), data.steps.size());
    if (std::fclose(file) != 0) ok = false;

    if (!ok) {
        std::remove(path.c_str());
        error = "Failed to write file: " + path;
        return false;
    }
    return true;
}

bool read_table_file(const std::string& path, TableData& data, std::string& error) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        error = "Cannot open file: " + path;
        return false;
    }

    std::uint8_t header[kFileHeaderSize];
    if (std::fread(header, 1, sizeof(header), file) != sizeof(header)) {
        std::fclose(file);
        error = "Incomplete file header: " + path;
        return false;
    }

    if (std::memcmp(header, kFileMagic, sizeof(kFileMagic)) != 0 ||
        get_u32le(header + 0x08) != kFileVersion ||
        get_u32le(header + 0x0C) != sizeof(index_t)) {
        std::fclose(file);
        error = "Unsupported jump-table file: " + path;
        return false;
    }

    const std::uint64_t table_size = get_u64le(header + 0x10);
    const std::uint64_t chain_count = get_u64le(header + 0x18);
    const std::uint64_t stored_checksum = get_u64le(header + 0x38);

    if (table_size < kMinTableSize || table_size > kMaxTableSize ||
        table_size % kChainDivisor != 0 || chain_count == 0 || chain_count > table_size) {
        std::fclose(file);
        error = "Invalid table size or chain count in file.";
        return false;
    }

    data.seed = get_u64le(header + 0x20);
    data.table_size = static_cast<index_t>(table_size);
    data.chain_count = static_cast<index_t>(chain_count);
    data.step_min = get_u32le(header + 0x28);
    data.step_max = get_u32le(header + 0x2C);

    try {
        data.table.resize(data.table_size);
        data.starts.resize(data.chain_count);
        data.steps.resize(data.chain_count);
    } catch (const std::bad_alloc&) {
        std::fclose(file);
        error = "Insufficient memory.";
        return false;
    }

    bool ok = read_indices(file, data.table.data(), data.table.size());
    ok = ok && read_indices(file, data.starts.data(), data.starts.size());
    ok = ok && read_indices(file, data.steps.data(), data.steps.size());
    const int extra_byte = std::fgetc(file);
    std::fclose(file);

    if (!ok || extra_byte != EOF) {
        error = "Incorrect jump-table file length.";
        return false;
    }

    std::uint64_t checksum = fnv1a(header, 0x38);
    checksum = hash_indices(data.table.data(), data.table.size(), checksum);
    checksum = hash_indices(data.starts.data(), data.starts.size(), checksum);
    checksum = hash_indices(data.steps.data(), data.steps.size(), checksum);
    if (checksum != stored_checksum) {
        error = "Jump-table file checksum mismatch.";
        return false;
    }

    if (!check_table_structure(data, error)) return false;
    for (index_t step : data.steps) {
        if (step < data.step_min || step > data.step_max || step == 0) {
            error = "Invalid step count in file.";
            return false;
        }
    }
    return true;
}

bool check_table_structure(const TableData& data, std::string& error) {
    std::vector<bool> seen(data.table_size, false);
    for (index_t value : data.table) {
        if (value >= data.table_size || seen[value]) {
            error = "Jump table is not a permutation of [0, N).";
            return false;
        }
        seen[value] = true;
    }

    std::vector<bool> seen_start(data.table_size, false);
    for (index_t start : data.starts) {
        if (start >= data.table_size || seen_start[start]) {
            error = "Chain start is out of range or duplicated.";
            return false;
        }
        seen_start[start] = true;
    }
    return true;
}

void chase_reference(const Problem& problem, index_t* final_pos) {
    for (index_t c = 0; c < problem.chain_count; ++c) {
        index_t index = problem.starts[c];
        for (index_t step = 0; step < problem.steps[c]; ++step) {
            index = problem.table[index];
        }
        final_pos[c] = index;
    }
}

std::uint64_t scan_table(const TableData& data) {
    std::uint64_t sum = 0;
    for (index_t value : data.table) sum += value;
    for (index_t value : data.starts) sum += value;
    for (index_t value : data.steps) sum += value;
    return sum;
}

}  // namespace chase
