# Histogram implementations

`src/histogram.h` exposes the existing `histogram(N, M, in)` entry point and
three independently callable implementations:

- `histogram_serial`: sequential counting.
- `histogram_private`: one padded private table per OpenMP worker, followed by
  a barrier and parallel merging of contiguous output blocks. Private storage
  has a 64 MiB budget by limiting workers (a direct call permits at least one row).
- `histogram_atomic`: a shared output table with OpenMP `atomic update relaxed`
  for every increment. The parallel-region barrier completes all updates before
  returning. No per-element mutex or stronger ordering is needed.

The dispatcher currently uses serial counting for N <= 32768 or a one-thread
configuration; otherwise it uses private tables for M <= 32768 and atomic
updates for larger M. These are initial policy constants in `src/histogram.cpp`,
not benchmark-tuned crossover points. Large-M hotspot inputs can still contend
heavily on the atomic path; this routing does not promise the fastest solution
for every distribution.

All implementations return an empty vector for M <= 0. Otherwise callers must
provide N >= 0, at least N input elements, and values in [0, M). Only the first
N elements are counted. N = 0 returns M zeros.

Build and run the original benchmark with `make` and
`OMP_NUM_THREADS=8 ./histogram_bench`. Run correctness checks with `make test`.
The tests compare every implementation and the dispatcher with the baseline,
including empty inputs, partial input ranges, uneven thread partitions,
dispatch boundaries, different distributions and dynamic OpenMP teams.
