#include "melspec.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

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

namespace {
struct FftPlan {
    bool vectorized = false;
    std::vector<int> reverse;
    std::vector<std::complex<double>> twiddles;

    explicit FftPlan(int n) {
        if (n <= 0 || (n & (n - 1)) != 0)
            throw std::invalid_argument("FFT length must be a positive power of two");
#if defined(__x86_64__) || defined(__i386__)
        vectorized = __builtin_cpu_supports("avx2");
#endif
        reverse.resize(n);
        for (int i = 1; i < n; ++i)
            reverse[i] = (reverse[i >> 1] >> 1) | ((i & 1) ? n >> 1 : 0);
        twiddles.resize(n - 1);
        for (int len = 2; len <= n;) {
            const int half = len / 2;
            for (int j = 0; j < half; ++j) {
                const double angle = -2.0 * M_PI * j / len;
                twiddles[half - 1 + j] = {std::cos(angle), std::sin(angle)};
            }
            if (len == n) break;
            len *= 2;
        }
    }
};

// Combine the first two radix-2 stages into four-point butterflies.
// The only nontrivial rotation is -i, implemented without multiplication.
void first_stages(std::complex<double>* a, int n) {
    if (n == 2) {
        const auto u = a[0];
        a[0] = u + a[1];
        a[1] = u - a[1];
    }
    for (int i = 0; i + 3 < n; i += 4) {
        const auto u0 = a[i] + a[i + 1];
        const auto u1 = a[i] - a[i + 1];
        const auto v0 = a[i + 2] + a[i + 3];
        const auto difference = a[i + 2] - a[i + 3];
        const std::complex<double> v1(difference.imag(), -difference.real());
        a[i] = u0 + v0;
        a[i + 1] = u1 + v1;
        a[i + 2] = u0 - v0;
        a[i + 3] = u1 - v1;
    }
}

void butterflies_scalar(std::complex<double>* a, int n, const FftPlan& plan) {
    for (int len = 8; len <= n;) {
        const int half = len / 2;
        const auto* weights = plan.twiddles.data() + half - 1;
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; ++j) {
                const auto u = a[i + j];
                const auto b = a[i + j + half];
                const auto w = weights[j];
                const std::complex<double> v(b.real() * w.real() - b.imag() * w.imag(),
                                             b.real() * w.imag() + b.imag() * w.real());
                a[i + j] = u + v;
                a[i + j + half] = u - v;
            }
        }
        if (len == n) break;
        len *= 2;
    }
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2")))
void butterflies_avx2(std::complex<double>* a, int n, const FftPlan& plan) {
    // std::complex<double> exposes interleaved real/imaginary double storage.
    auto* data = reinterpret_cast<double*>(a);
    for (int len = 8; len <= n;) {
        const int half = len / 2;
        const auto* weights = reinterpret_cast<const double*>(
            plan.twiddles.data() + half - 1);
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; j += 2) {
                const __m256d u = _mm256_loadu_pd(data + 2 * (i + j));
                const __m256d b = _mm256_loadu_pd(data + 2 * (i + j + half));
                const __m256d w = _mm256_loadu_pd(weights + 2 * j);
                const __m256d wr = _mm256_movedup_pd(w);
                const __m256d wi = _mm256_permute_pd(w, 0xf);
                const __m256d swapped = _mm256_permute_pd(b, 0x5);
                const __m256d v = _mm256_addsub_pd(_mm256_mul_pd(b, wr),
                                                   _mm256_mul_pd(swapped, wi));
                _mm256_storeu_pd(data + 2 * (i + j), _mm256_add_pd(u, v));
                _mm256_storeu_pd(data + 2 * (i + j + half), _mm256_sub_pd(u, v));
            }
        }
        if (len == n) break;
        len *= 2;
    }
}
#endif

void fft_inplace(std::vector<std::complex<double>>& a, const FftPlan& plan) {
    const int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i)
        if (i < plan.reverse[i]) std::swap(a[i], a[plan.reverse[i]]);
    first_stages(a.data(), n);
#if defined(__x86_64__) || defined(__i386__)
    if (plan.vectorized) {
        butterflies_avx2(a.data(), n, plan);
        return;
    }
#endif
    butterflies_scalar(a.data(), n, plan);
}

struct RealFftPlan {
    int n;
    FftPlan packed_plan;
    std::vector<std::complex<double>> recovery;

    explicit RealFftPlan(int length)
        : n(length), packed_plan(length > 1 ? length / 2 : 1) {
        if (n <= 0 || (n & (n - 1)) != 0)
            throw std::invalid_argument("FFT length must be a positive power of two");
        recovery.resize(n / 2 + 1);
        for (int k = 0; k <= n / 2; ++k) {
            const double angle = -2.0 * M_PI * k / n;
            recovery[k] = {std::cos(angle), std::sin(angle)};
        }
    }
};

std::complex<double> real_bin(const std::vector<std::complex<double>>& packed,
                              const RealFftPlan& plan, int k) {
    if (plan.n == 1) return {packed[0].real(), 0.0};
    if (k == 0) return {packed[0].real() + packed[0].imag(), 0.0};
    if (k == plan.n / 2)
        return {packed[0].real() - packed[0].imag(), 0.0};
    const auto a = packed[k];
    const auto b = std::conj(packed[plan.n / 2 - k]);
    const auto sum = a + b;
    const auto difference = (a - b) * plan.recovery[k];
    // X[k] = ((a+b) - i*W[k]*(a-b)) / 2.
    return {0.5 * (sum.real() + difference.imag()),
            0.5 * (sum.imag() - difference.real())};
}
}  // namespace

// Pack even/odd real samples into a half-length complex FFT.
void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    const int n = static_cast<int>(frame.size());
    const RealFftPlan plan(n);
    std::vector<std::complex<double>> packed(std::max(1, n / 2));
    if (n == 1) {
        packed[0] = {frame[0], 0.0};
    } else {
        for (int i = 0; i < n / 2; ++i)
            packed[i] = {frame[2 * i], frame[2 * i + 1]};
    }
    fft_inplace(packed, plan.packed_plan);
    out.resize(n / 2 + 1);
    for (int k = 0; k <= n / 2; ++k) out[k] = real_bin(packed, plan, k);
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

namespace {
struct MelBand {
    int first;
    std::vector<double> weights;
};

std::vector<MelBand> sparse_filterbank(int sr, int n_fft, int n_mels,
                                      double f_min, double f_max) {
    const auto dense = build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);
    std::vector<MelBand> bands;
    bands.reserve(n_mels);
    for (const auto& weights : dense) {
        int first = 0;
        int end = static_cast<int>(weights.size());
        while (first < end && weights[first] == 0.0) ++first;
        while (end > first && weights[end - 1] == 0.0) --end;
        bands.push_back({first, std::vector<double>(weights.begin() + first,
                                                   weights.begin() + end)});
    }
    return bands;
}
double dot_scalar(const double* weights, const double* power, std::size_t count) {
    double sum = 0.0;
    for (std::size_t k = 0; k < count; ++k) sum += weights[k] * power[k];
    return sum;
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2")))
double dot_avx2(const double* weights, const double* power, std::size_t count) {
    __m256d sum = _mm256_setzero_pd();
    std::size_t k = 0;
    for (; k + 4 <= count; k += 4) {
        sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(weights + k),
                                              _mm256_loadu_pd(power + k)));
    }
    const __m128d pair = _mm_add_pd(_mm256_castpd256_pd128(sum),
                                   _mm256_extractf128_pd(sum, 1));
    double result = _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));
    for (; k < count; ++k) result += weights[k] * power[k];
    return result;
}
#endif
}  // namespace

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

    const auto mel_fb = sparse_filterbank(sr, n_fft, n_mels, f_min, f_max);
    const RealFftPlan fft_plan(n_fft);
    auto dot = &dot_scalar;
#if defined(__x86_64__) || defined(__i386__)
    if (fft_plan.packed_plan.vectorized) dot = &dot_avx2;
#endif
    output.resize(static_cast<std::size_t>(n_mels) * n_frames);
    const unsigned int available = std::max(1u, std::thread::hardware_concurrency());
    const int workers = std::min({16, static_cast<int>(std::min(available, 16u)),
                                  1 + (n_frames - 1) / 64});
    // Allocate all thread-private buffers before launching any worker.
    std::vector<std::vector<std::complex<double>>> fft_buffers(
        workers, std::vector<std::complex<double>>(std::max(1, n_fft / 2)));
    std::vector<std::vector<double>> power_buffers(
        workers, std::vector<double>(n_bins));

    const auto process_frames = [&](int worker) {
        auto& fft_data = fft_buffers[worker];
        auto& power = power_buffers[worker];
        const int first = static_cast<long long>(n_frames) * worker / workers;
        const int end = static_cast<long long>(n_frames) * (worker + 1) / workers;
        for (int t = first; t < end; ++t) {
            const std::size_t start = static_cast<std::size_t>(t) * hop_length;
            if (n_fft == 1) {
                fft_data[0] = {y[start] * window[0], 0.0};
            } else {
                for (int i = 0; i < n_fft / 2; ++i)
                    fft_data[i] = {y[start + 2 * i] * window[2 * i],
                                   y[start + 2 * i + 1] * window[2 * i + 1]};
            }
            fft_inplace(fft_data, fft_plan.packed_plan);

            for (int k = 0; k < n_bins; ++k) {
                const auto value = real_bin(fft_data, fft_plan, k);
                power[k] = value.real() * value.real() + value.imag() * value.imag();
            }

            for (int m = 0; m < n_mels; ++m) {
                const auto& band = mel_fb[m];
                const double acc = dot(band.weights.data(), power.data() + band.first,
                                       band.weights.size());
                output[static_cast<std::size_t>(m) * n_frames + t] = acc;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    try {
        for (int worker = 1; worker < workers; ++worker)
            threads.emplace_back(process_frames, worker);
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    process_frames(0);
    for (auto& thread : threads) thread.join();
}
