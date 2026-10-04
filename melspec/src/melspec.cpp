#include "melspec.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
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
    std::vector<std::pair<int, int>> swaps;
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
        for (int i = 1; i < n; ++i)
            if (i < reverse[i]) swaps.emplace_back(i, reverse[i]);
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
inline __m256d multiply_complex_pair(__m256d b, __m256d w) {
    const __m256d wr = _mm256_movedup_pd(w);
    const __m256d wi = _mm256_permute_pd(w, 0xf);
    const __m256d swapped = _mm256_permute_pd(b, 0x5);
    return _mm256_addsub_pd(_mm256_mul_pd(b, wr), _mm256_mul_pd(swapped, wi));
}

__attribute__((target("avx2")))
void butterflies_avx2(std::complex<double>* a, int n, const FftPlan& plan) {
    // std::complex<double> exposes interleaved real/imaginary double storage.
    auto* data = reinterpret_cast<double*>(a);
    for (int len = 8; len <= n;) {
        const int half = len / 2;
        const auto* weights = reinterpret_cast<const double*>(
            plan.twiddles.data() + half - 1);
        // Fuse two stages: keep the first-stage results in registers instead
        // of storing and loading the entire array between stages.
        if (len <= n / 2) {
            const auto* next_weights = reinterpret_cast<const double*>(
                plan.twiddles.data() + len - 1);
            for (int i = 0; i < n; i += 2 * len) {
                for (int j = 0; j < half; j += 2) {
                    double* p = data + 2 * (i + j);
                    const __m256d w = _mm256_loadu_pd(weights + 2 * j);
                    const __m256d a0 = _mm256_loadu_pd(p);
                    const __m256d a1 = multiply_complex_pair(
                        _mm256_loadu_pd(p + 2 * half), w);
                    const __m256d a2 = _mm256_loadu_pd(p + 2 * len);
                    const __m256d a3 = multiply_complex_pair(
                        _mm256_loadu_pd(p + 2 * (len + half)), w);
                    const __m256d u0 = _mm256_add_pd(a0, a1);
                    const __m256d u1 = _mm256_sub_pd(a0, a1);
                    const __m256d next_w = _mm256_loadu_pd(next_weights + 2 * j);
                    const __m256d v0 = multiply_complex_pair(_mm256_add_pd(a2, a3), next_w);
                    const __m256d rotated = multiply_complex_pair(_mm256_sub_pd(a2, a3), next_w);
                    // W[j + half] = -i * W[j] in the second fused stage.
                    const __m256d v1 = _mm256_xor_pd(_mm256_permute_pd(rotated, 0x5),
                                                    _mm256_set_pd(-0.0, 0.0, -0.0, 0.0));
                    _mm256_storeu_pd(p, _mm256_add_pd(u0, v0));
                    _mm256_storeu_pd(p + 2 * half, _mm256_add_pd(u1, v1));
                    _mm256_storeu_pd(p + 2 * len, _mm256_sub_pd(u0, v0));
                    _mm256_storeu_pd(p + 2 * (len + half), _mm256_sub_pd(u1, v1));
                }
            }
            if (len == n / 2) break;
            len *= 4;
            continue;
        }
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
    for (const auto& indices : plan.swaps)
        std::swap(a[indices.first], a[indices.second]);
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

// Separate endpoint handling from the uniform recovery/power loop so the
// compiler can vectorize interior bins without per-bin special cases.
void compute_power(const std::complex<double>* packed, const RealFftPlan& plan,
                   double* power, int first, int end) {
    // Only [first, end) is consumed by the sparse Mel projection.
    if (first >= end) return;
    if (plan.n == 1) {
        const double value = packed[0].real();
        power[0] = value * value;
        return;
    }

    const int half = plan.n / 2;
    if (first == 0) {
        const double dc = packed[0].real() + packed[0].imag();
        power[0] = dc * dc;
    }
    if (end > half) {
        const double nyquist = packed[0].real() - packed[0].imag();
        power[half] = nyquist * nyquist;
    }

    const int begin = std::max(first, 1);
    const int finish = std::min(end, half);
    const auto single_range = [&](int begin, int finish) {
        for (int k = begin; k < finish; ++k) {
            const double ar = packed[k].real();
            const double ai = packed[k].imag();
            const double br = packed[half - k].real();
            const double bi = packed[half - k].imag();
            const double wr = plan.recovery[k].real();
            const double wi = plan.recovery[k].imag();
            // b is conjugated: a+b = (ar+br, ai-bi), a-b = (ar-br, ai+bi).
            const double dr = ar - br;
            const double di = ai + bi;
            const double rotated_r = dr * wr - di * wi;
            const double rotated_i = dr * wi + di * wr;
            const double real = 0.5 * ((ar + br) + rotated_i);
            const double imag = 0.5 * ((ai - bi) - rotated_r);
            power[k] = real * real + imag * imag;
        }
    };
    // Pair only bins whose two outputs are both used by Mel projection.
    const int pair_begin = std::max(begin, half - finish + 1);
    const int pair_end = std::min({finish, half - begin + 1, (half + 1) / 2});
    if (pair_begin >= pair_end) {
        single_range(begin, finish);
        return;
    }
    single_range(begin, pair_begin);
    single_range(pair_end, half - pair_end + 1);
    single_range(half - pair_begin + 1, finish);
    for (int k = pair_begin; k < pair_end; ++k) {
        const double ar = packed[k].real();
        const double ai = packed[k].imag();
        const double br = packed[half - k].real();
        const double bi = packed[half - k].imag();
        const double wr = plan.recovery[k].real();
        const double wi = plan.recovery[k].imag();
        const double sr = ar + br;
        const double si = ai - bi;
        const double dr = ar - br;
        const double di = ai + bi;
        const double rr = dr * wr - di * wi;
        const double ri = dr * wi + di * wr;
        const double real0 = 0.5 * (sr + ri);
        const double imag0 = 0.5 * (si - rr);
        const double real1 = 0.5 * (sr - ri);
        const double imag1 = -0.5 * (si + rr);
        power[k] = real0 * real0 + imag0 * imag0;
        power[half - k] = real1 * real1 + imag1 * imag1;
    }
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
    const int n_bins = n_fft / 2 + 1;
    std::vector<double> frequencies(n_bins);
    for (int k = 0; k < n_bins; ++k)
        frequencies[k] = static_cast<double>(k) * sr / static_cast<double>(n_fft);
    const double mel_min = hz_to_mel(f_min);
    const double mel_max = hz_to_mel(f_max);
    std::vector<double> hz_points(n_mels + 2);
    for (int i = 0; i <= n_mels + 1; ++i)
        hz_points[i] = mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1));

    std::vector<MelBand> bands;
    bands.reserve(n_mels);
    for (int m = 0; m < n_mels; ++m) {
        const double left = hz_points[m];
        const double center = hz_points[m + 1];
        const double right = hz_points[m + 2];
        const double norm = 2.0 / (right - left);
        int first = 0;
        int end = n_bins;
        // Include both endpoints, then evaluate/trim with the original formula.
        // Degenerate triangles keep the original full-bin arithmetic.
        if (sr > 0 && left < center && center < right && std::isfinite(norm)
            && std::isfinite(left) && std::isfinite(right)) {
            first = static_cast<int>(std::lower_bound(frequencies.begin(), frequencies.end(), left)
                                     - frequencies.begin());
            end = static_cast<int>(std::upper_bound(frequencies.begin(), frequencies.end(), right)
                                   - frequencies.begin());
        }
        std::vector<double> weights(end - first);
        for (int k = first; k < end; ++k) {
            const double rising = (frequencies[k] - left) / (center - left);
            const double falling = (right - frequencies[k]) / (right - center);
            weights[k - first] = std::max(0.0, std::min(rising, falling)) * norm;
        }
        std::size_t begin = 0;
        std::size_t finish = weights.size();
        while (begin < finish && weights[begin] == 0.0) ++begin;
        while (finish > begin && weights[finish - 1] == 0.0) --finish;
        if (begin == finish) {
            bands.push_back({n_bins, {}});
        } else {
            if (begin != 0)
                std::move(weights.begin() + begin, weights.begin() + finish, weights.begin());
            weights.resize(finish - begin);
            bands.push_back({first + static_cast<int>(begin), std::move(weights)});
        }
    }
    return bands;
}
// Dispatch once per frame, keeping all band dot products inside this function.
void project_mel_scalar(const std::vector<MelBand>& bands, const double* power,
                        double* output, std::size_t n_frames, std::size_t frame) {
    for (std::size_t m = 0; m < bands.size(); ++m) {
        const auto& band = bands[m];
        const double* values = power + band.first;
        double sum = 0.0;
        for (std::size_t k = 0; k < band.weights.size(); ++k)
            sum += band.weights[k] * values[k];
        output[m * n_frames + frame] = sum;
    }
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2")))
void project_mel_avx2(const std::vector<MelBand>& bands, const double* power,
                      double* output, std::size_t n_frames, std::size_t frame) {
    for (std::size_t m = 0; m < bands.size(); ++m) {
        const auto& band = bands[m];
        const double* weights = band.weights.data();
        const double* values = power + band.first;
        const std::size_t count = band.weights.size();
        double result = 0.0;
        std::size_t k = 0;
        // Fewer than four doubles cannot fill one AVX2 vector; skip its reduction.
        if (count >= 4) {
            __m256d sum = _mm256_setzero_pd();
            if (count >= 16) {
                __m256d sum1 = _mm256_setzero_pd();
                __m256d sum2 = _mm256_setzero_pd();
                __m256d sum3 = _mm256_setzero_pd();
                for (; k + 16 <= count; k += 16) {
                    sum = _mm256_add_pd(sum, _mm256_mul_pd(
                        _mm256_loadu_pd(weights + k), _mm256_loadu_pd(values + k)));
                    sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(
                        _mm256_loadu_pd(weights + k + 4), _mm256_loadu_pd(values + k + 4)));
                    sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(
                        _mm256_loadu_pd(weights + k + 8), _mm256_loadu_pd(values + k + 8)));
                    sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(
                        _mm256_loadu_pd(weights + k + 12), _mm256_loadu_pd(values + k + 12)));
                }
                sum = _mm256_add_pd(_mm256_add_pd(sum, sum1), _mm256_add_pd(sum2, sum3));
            }
            for (; k + 4 <= count; k += 4) {
                sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_loadu_pd(weights + k),
                                                      _mm256_loadu_pd(values + k)));
            }
            const __m128d pair = _mm_add_pd(_mm256_castpd256_pd128(sum),
                                           _mm256_extractf128_pd(sum, 1));
            result = _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));
        }
        for (; k < count; ++k) result += weights[k] * values[k];
        output[m * n_frames + frame] = result;
    }
}

// Each vector lane carries one independent time frame.
struct alignas(32) FrameBatchComplex {
    double real[4];
    double imag[4];
};

__attribute__((target("avx2")))
void compute_four_frames(const double* samples, const double* window, int hop,
                         const RealFftPlan& plan, const std::vector<MelBand>& bands,
                         int power_first, int power_end, FrameBatchComplex* packed,
                         double* power, double* output, std::size_t output_stride) {
    const int n = plan.n;
    const int half = n / 2;
    const int packed_size = std::max(1, half);
    if (n == 1) {
        for (int lane = 0; lane < 4; ++lane) {
            packed[0].real[lane] = samples[static_cast<std::size_t>(lane) * hop] * window[0];
            packed[0].imag[lane] = 0.0;
        }
    } else {
        for (int k = 0; k < half; ++k) {
            for (int lane = 0; lane < 4; ++lane) {
                const double* frame = samples + static_cast<std::size_t>(lane) * hop;
                packed[k].real[lane] = frame[2 * k] * window[2 * k];
                packed[k].imag[lane] = frame[2 * k + 1] * window[2 * k + 1];
            }
        }
    }
    for (const auto& indices : plan.packed_plan.swaps)
        std::swap(packed[indices.first], packed[indices.second]);
    for (int len = 2; len <= packed_size;) {
        const int h = len / 2;
        const auto* weights = plan.packed_plan.twiddles.data() + h - 1;
        for (int i = 0; i < packed_size; i += len) {
            for (int j = 0; j < h; ++j) {
                auto& a = packed[i + j];
                auto& b = packed[i + j + h];
                const __m256d ar = _mm256_load_pd(a.real);
                const __m256d ai = _mm256_load_pd(a.imag);
                const __m256d br = _mm256_load_pd(b.real);
                const __m256d bi = _mm256_load_pd(b.imag);
                const __m256d wr = _mm256_set1_pd(weights[j].real());
                const __m256d wi = _mm256_set1_pd(weights[j].imag());
                const __m256d vr = _mm256_sub_pd(_mm256_mul_pd(br, wr), _mm256_mul_pd(bi, wi));
                const __m256d vi = _mm256_add_pd(_mm256_mul_pd(br, wi), _mm256_mul_pd(bi, wr));
                _mm256_store_pd(a.real, _mm256_add_pd(ar, vr));
                _mm256_store_pd(a.imag, _mm256_add_pd(ai, vi));
                _mm256_store_pd(b.real, _mm256_sub_pd(ar, vr));
                _mm256_store_pd(b.imag, _mm256_sub_pd(ai, vi));
            }
        }
        if (len == packed_size) break;
        len *= 2;
    }
    if (power_first < power_end) {
        const __m256d r0 = _mm256_load_pd(packed[0].real);
        const __m256d i0 = _mm256_load_pd(packed[0].imag);
        if (power_first == 0) {
            const __m256d dc = n == 1 ? r0 : _mm256_add_pd(r0, i0);
            _mm256_storeu_pd(power, _mm256_mul_pd(dc, dc));
        }
        if (n > 1 && power_end > half) {
            const __m256d nyquist = _mm256_sub_pd(r0, i0);
            _mm256_storeu_pd(power + static_cast<std::size_t>(half) * 4,
                             _mm256_mul_pd(nyquist, nyquist));
        }
        const __m256d scale = _mm256_set1_pd(0.5);
        const int end = std::min(power_end, half);
        for (int k = std::max(1, power_first); k < end; ++k) {
            const __m256d ar = _mm256_load_pd(packed[k].real);
            const __m256d ai = _mm256_load_pd(packed[k].imag);
            const __m256d br = _mm256_load_pd(packed[half - k].real);
            const __m256d bi = _mm256_load_pd(packed[half - k].imag);
            const __m256d wr = _mm256_set1_pd(plan.recovery[k].real());
            const __m256d wi = _mm256_set1_pd(plan.recovery[k].imag());
            const __m256d dr = _mm256_sub_pd(ar, br);
            const __m256d di = _mm256_add_pd(ai, bi);
            const __m256d rr = _mm256_sub_pd(_mm256_mul_pd(dr, wr), _mm256_mul_pd(di, wi));
            const __m256d ri = _mm256_add_pd(_mm256_mul_pd(dr, wi), _mm256_mul_pd(di, wr));
            const __m256d real = _mm256_mul_pd(scale, _mm256_add_pd(_mm256_add_pd(ar, br), ri));
            const __m256d imag = _mm256_mul_pd(scale, _mm256_sub_pd(_mm256_sub_pd(ai, bi), rr));
            _mm256_storeu_pd(power + static_cast<std::size_t>(k) * 4,
                             _mm256_add_pd(_mm256_mul_pd(real, real), _mm256_mul_pd(imag, imag)));
        }
    }
    for (std::size_t m = 0; m < bands.size(); ++m) {
        const auto& band = bands[m];
        __m256d sum = _mm256_setzero_pd();
        for (std::size_t k = 0; k < band.weights.size(); ++k) {
            const __m256d weight = _mm256_set1_pd(band.weights[k]);
            const __m256d values = _mm256_loadu_pd(power + (band.first + k) * 4);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(weight, values));
        }
        _mm256_storeu_pd(output + m * output_stride, sum);
    }
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
    // Cover every nonempty band; bins outside this range are never read.
    // Keep the complete packed FFT for the conjugate pairs used in recovery.
    int power_first = n_bins;
    int power_end = 0;
    for (const auto& band : mel_fb) {
        if (band.weights.empty()) continue;
        power_first = std::min(power_first, band.first);
        power_end = std::max(power_end,
                             band.first + static_cast<int>(band.weights.size()));
    }
    const RealFftPlan fft_plan(n_fft);
    auto project_mel = &project_mel_scalar;
#if defined(__x86_64__) || defined(__i386__)
    if (fft_plan.packed_plan.vectorized) project_mel = &project_mel_avx2;
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

    // Eight doubles form one 64-byte output segment per Mel band.
    constexpr int frame_block = 8;
    std::vector<std::vector<double>> mel_buffers(
        workers, std::vector<double>(static_cast<std::size_t>(n_mels) * frame_block));

#if defined(__x86_64__) || defined(__i386__)
    const bool batch_frames = fft_plan.packed_plan.vectorized && n_frames / workers >= 4;
    std::vector<std::vector<FrameBatchComplex>> batch_fft_buffers(
        workers, std::vector<FrameBatchComplex>(batch_frames ? std::max(1, n_fft / 2) : 0));
    std::vector<std::vector<double>> batch_power_buffers(
        workers, std::vector<double>(batch_frames ? static_cast<std::size_t>(n_bins) * 4 : 0));
#endif

    const auto process_frames = [&](int worker) {
        auto& fft_data = fft_buffers[worker];
        auto& power = power_buffers[worker];
        auto& mel_block = mel_buffers[worker];
        const int first = static_cast<long long>(n_frames) * worker / workers;
        const int end = static_cast<long long>(n_frames) * (worker + 1) / workers;
        for (int block = first; block < end;) {
            const int count = std::min(frame_block, end - block);
            int lane = 0;
#if defined(__x86_64__) || defined(__i386__)
            if (batch_frames) {
                for (; lane + 4 <= count; lane += 4)
                    compute_four_frames(y.data() + static_cast<std::size_t>(block + lane) * hop_length,
                                        window.data(), hop_length, fft_plan, mel_fb,
                                        power_first, power_end, batch_fft_buffers[worker].data(),
                                        batch_power_buffers[worker].data(), mel_block.data() + lane,
                                        frame_block);
            }
#endif
            for (; lane < count; ++lane) {
                const int t = block + lane;
                const std::size_t start = static_cast<std::size_t>(t) * hop_length;
                if (n_fft == 1) {
                    fft_data[0] = {y[start] * window[0], 0.0};
                } else {
                    for (int i = 0; i < n_fft / 2; ++i)
                        fft_data[i] = {y[start + 2 * i] * window[2 * i],
                                       y[start + 2 * i + 1] * window[2 * i + 1]};
                }
                fft_inplace(fft_data, fft_plan.packed_plan);

                compute_power(fft_data.data(), fft_plan, power.data(),
                              power_first, power_end);

                project_mel(mel_fb, power.data(), mel_block.data(), frame_block, lane);
            }
            for (int m = 0; m < n_mels; ++m)
                std::copy_n(mel_block.data() + static_cast<std::size_t>(m) * frame_block,
                            count, output.data() + static_cast<std::size_t>(m) * n_frames + block);
            block += count;
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
