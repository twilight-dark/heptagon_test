#pragma once
#include <complex>
#include <string>
#include <vector>

// HTK mel scale conversions
double hz_to_mel(double f);
double mel_to_hz(double m);

// Cooley-Tukey iterative real FFT.
// Input: real vector of length n (power of 2).
// Output: complex vector of length n/2+1 (DC + positive frequencies).
void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out);

// Build HTK mel filterbank with Slaney area normalization.
// Returns 2D vector [n_mels][n_fft/2+1].
std::vector<std::vector<double>> build_mel_filterbank(int sr, int n_fft,
                                                      int n_mels, double f_min,
                                                      double f_max);

// Compute mel spectrogram. All calculations in double.
// Output: flat row-major vector of shape (n_mels, n_frames), double.
void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output);

// Read int16 LE PCM file, normalize by /32768.0, store as double.
void read_pcm(const std::string& path, std::vector<double>& y);

// Write row-major binary (casts double to float32 LE at output).
void write_output(const std::string& path, const std::vector<double>& data,
                  int n_mels, int n_frames);
