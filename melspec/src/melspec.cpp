#include "melspec.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// HTK mel scale
// ─────────────────────────────────────────────────────────────────────────────

double hz_to_mel(double f) {
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

double mel_to_hz(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cooley-Tukey iterative FFT (in-place, complex input/output)
// ─────────────────────────────────────────────────────────────────────────────

static void fft_inplace(std::vector<std::complex<double>>& a) {
    const int n = static_cast<int>(a.size());
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    // Cooley-Tukey butterfly stages
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / static_cast<double>(len);
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Real FFT: input n real values → output n/2+1 complex values.
void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    const int n = static_cast<int>(frame.size());
    // Copy real input into complex array
    std::vector<std::complex<double>> a(n);
    for (int i = 0; i < n; ++i) a[i] = {frame[i], 0.0};
    fft_inplace(a);
    // Keep only DC + positive frequencies (n/2+1 bins)
    out.resize(n / 2 + 1);
    for (int k = 0; k <= n / 2; ++k) out[k] = a[k];
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel filterbank
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::vector<double>> build_mel_filterbank(int sr, int n_fft,
                                                      int n_mels, double f_min,
                                                      double f_max) {
    const int n_bins = n_fft / 2 + 1;

    // FFT bin center frequencies
    std::vector<double> fft_freqs(n_bins);
    for (int k = 0; k < n_bins; ++k)
        fft_freqs[k] = static_cast<double>(k) * sr / static_cast<double>(n_fft);

    // n_mels+2 equally-spaced mel points
    double mel_min = hz_to_mel(f_min);
    double mel_max = hz_to_mel(f_max);
    std::vector<double> hz_points(n_mels + 2);
    for (int i = 0; i <= n_mels + 1; ++i)
        hz_points[i] =
            mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1));

    // Build triangular filters with Slaney area normalization
    std::vector<std::vector<double>> weights(n_mels,
                                             std::vector<double>(n_bins, 0.0));

    for (int m = 0; m < n_mels; ++m) {
        double f_l = hz_points[m];
        double f_c = hz_points[m + 1];
        double f_r = hz_points[m + 2];
        double enorm = 2.0 / (f_r - f_l);

        for (int k = 0; k < n_bins; ++k) {
            double rising = (fft_freqs[k] - f_l) / (f_c - f_l);
            double falling = (f_r - fft_freqs[k]) / (f_r - f_c);
            double val = std::max(0.0, std::min(rising, falling));
            weights[m][k] = val * enorm;
        }
    }
    return weights;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel spectrogram computation
// ─────────────────────────────────────────────────────────────────────────────

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output) {
    const int n_samples = static_cast<int>(y.size());
    const int n_frames = 1 + (n_samples - n_fft) / hop_length;
    const int n_bins = n_fft / 2 + 1;

    if (n_frames <= 0)
        throw std::runtime_error("Audio too short for given n_fft");

    // Periodic Hann window: w[n] = 0.5 * (1 - cos(2*pi*n/n_fft))
    std::vector<double> window(n_fft);
    for (int i = 0; i < n_fft; ++i)
        window[i] =
            0.5 * (1.0 - std::cos(2.0 * M_PI * i / static_cast<double>(n_fft)));

    // Power spectrum: power_spec[k][t], shape (n_bins, n_frames)
    std::vector<std::vector<double>> power_spec(n_bins,
                                                std::vector<double>(n_frames));

    for (int t = 0; t < n_frames; ++t) {
        int start = t * hop_length;

        // Window the frame
        std::vector<double> frame(n_fft);
        for (int i = 0; i < n_fft; ++i) frame[i] = y[start + i] * window[i];

        // FFT
        std::vector<std::complex<double>> fft_out;
        compute_rfft(frame, fft_out);

        // |X|^2
        for (int k = 0; k < n_bins; ++k) {
            double re = fft_out[k].real();
            double im = fft_out[k].imag();
            power_spec[k][t] = re * re + im * im;
        }
    }

    // Build mel filterbank: mel_fb[m][k]
    auto mel_fb = build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);

    // Apply filterbank: output[m][t] = sum_k mel_fb[m][k] * power_spec[k][t]
    output.resize(static_cast<size_t>(n_mels) * n_frames);
    for (int m = 0; m < n_mels; ++m) {
        for (int t = 0; t < n_frames; ++t) {
            double acc = 0.0;
            for (int k = 0; k < n_bins; ++k)
                acc += mel_fb[m][k] * power_spec[k][t];
            output[m * n_frames + t] = acc;
        }
    }
}
