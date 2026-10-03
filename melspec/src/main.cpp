/**
 * melspec.cpp — C++ mel spectrogram reference implementation.
 *
 * Matches librosa.feature.melspectrogram(y=y, sr=sr, center=False, htk=True).
 * All DSP in double precision; output cast to float32 only at the end.
 *
 * Build:
 *   mkdir -p ref/cpp/build && cd ref/cpp/build && cmake .. && make
 *
 * Usage:
 *   ./melspec <input_pcm> <output_bin> [--sr SR] [--n_fft N_FFT]
 *             [--hop_length HOP] [--n_mels N_MELS] [--f_min F_MIN]
 *             [--f_max F_MAX]
 */
#include "melspec.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// I/O
// ─────────────────────────────────────────────────────────────────────────────

void read_pcm(const std::string& path, std::vector<double>& y) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open input: " + path);

    f.seekg(0, std::ios::end);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);

    int n_samples = static_cast<int>(size / sizeof(int16_t));
    std::vector<int16_t> buf(n_samples);
    if (!f.read(reinterpret_cast<char*>(buf.data()), size))
        throw std::runtime_error("Failed to read PCM: " + path);

    y.resize(n_samples);
    for (int i = 0; i < n_samples; ++i)
        y[i] = static_cast<double>(buf[i]) / 32768.0;
}

void write_output(const std::string& path, const std::vector<double>& data,
                  int n_mels, int n_frames) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open output: " + path);

    // Cast double to float32 only at output
    std::vector<float> out_f32(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        out_f32[i] = static_cast<float>(data[i]);

    f.write(reinterpret_cast<const char*>(out_f32.data()),
            static_cast<std::streamsize>(out_f32.size() * sizeof(float)));
    if (!f) throw std::runtime_error("Failed to write output: " + path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple argument parser
// ─────────────────────────────────────────────────────────────────────────────

struct Args {
    std::string input_pcm;
    std::string output_bin;
    int sr = 44100;
    int n_fft = 2048;
    int hop_length = 512;
    int n_mels = 128;
    double f_min = 0.0;
    double f_max = 22050.0;
};

static void print_help(const char* prog) {
    std::cout
        << "Usage: " << prog
        << " <input_pcm> <output_bin>"
           " [--sr SR] [--n_fft N_FFT] [--hop_length HOP]"
           " [--n_mels N_MELS] [--f_min F_MIN] [--f_max F_MAX]\n\n"
        << "  input_pcm    Input PCM file (int16 LE mono)\n"
        << "  output_bin   Output binary file (float32 LE row-major [n_mels, "
           "n_frames])\n"
        << "  --sr         Sample rate in Hz (default: 44100)\n"
        << "  --n_fft      FFT size, power of 2 (default: 2048)\n"
        << "  --hop_length Samples between successive frames (default: 512)\n"
        << "  --n_mels     Number of Mel bands (default: 128)\n"
        << "  --f_min      Lowest frequency in Hz (default: 0.0)\n"
        << "  --f_max      Highest frequency in Hz (default: 22050.0)\n";
}

static Args parse_args(int argc, char** argv) {
    if (argc < 3) {
        if (argc == 2 && (std::string(argv[1]) == "--help" ||
                          std::string(argv[1]) == "-h")) {
            print_help(argv[0]);
            std::exit(0);
        }
        print_help(argv[0]);
        std::exit(1);
    }

    Args a;
    a.input_pcm = argv[1];
    a.output_bin = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string key(argv[i]);
        if (key == "--help" || key == "-h") {
            print_help(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << key << "\n";
            std::exit(1);
        }
        std::string val(argv[++i]);
        if (key == "--sr")
            a.sr = std::stoi(val);
        else if (key == "--n_fft")
            a.n_fft = std::stoi(val);
        else if (key == "--hop_length")
            a.hop_length = std::stoi(val);
        else if (key == "--n_mels")
            a.n_mels = std::stoi(val);
        else if (key == "--f_min")
            a.f_min = std::stod(val);
        else if (key == "--f_max")
            a.f_max = std::stod(val);
        else {
            std::cerr << "Unknown argument: " << key << "\n";
            std::exit(1);
        }
    }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    try {
        Args a = parse_args(argc, argv);

        std::vector<double> y;
        read_pcm(a.input_pcm, y);

        std::vector<double> output;
        compute_melspectrogram(y, a.sr, a.n_fft, a.hop_length, a.n_mels,
                               a.f_min, a.f_max, output);

        int n_frames =
            1 + (static_cast<int>(y.size()) - a.n_fft) / a.hop_length;
        write_output(a.output_bin, output, a.n_mels, n_frames);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
