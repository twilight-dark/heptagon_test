# Histogram implementations

`src/histogram.cpp` contains the existing `histogram(N, M, in)` entry point and
three implementations, without a project header dependency:

- `histogram_serial`: sequential counting.
- `histogram_private`: one padded private table per OpenMP worker, followed by
  a barrier and parallel merging of contiguous output blocks. Private storage
  has a 64 MiB budget by limiting workers (a direct call permits at least one row).
- `histogram_atomic`: each worker aggregates counts in a fixed-size private hash
  table (1024 slots, at most 512 distinct values, about 12 KiB on a 64-bit host).
  Existing values keep accumulating locally. When a new value arrives at capacity,
  occupied entries are merged into the shared output with OpenMP
  `atomic update relaxed`, then cleared. Each worker also flushes its final partial
  batch. The parallel-region barrier completes all updates before returning.
  This reduces atomic operations for repeated values without allocating M private
  counters per worker. Uniform input may pay extra hashing cost with little aggregation.

The dispatcher currently uses serial counting for N <= 32768 or a one-thread
configuration; otherwise it uses private tables for M <= 32768 and atomic
updates for larger M. These are initial policy constants in `src/histogram.cpp`,
not benchmark-tuned crossover points. Hotspot sets larger than the small table can
still cause frequent flushes and contention; this routing does not promise the
fastest solution for every distribution. No distribution detection is performed.

All implementations return an empty vector for M <= 0. Otherwise callers must
provide N >= 0, at least N input elements, and values in [0, M). Only the first
N elements are counted. N = 0 returns M zeros.

Build and run the original benchmark with `make` and
`OMP_NUM_THREADS=8 ./histogram_bench`. The benchmark compares the dispatcher
with the original serial baseline across its 12 cases.
