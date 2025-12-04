# Cross-Language Benchmark

Performance comparison between Python, Go, and Rust implementations of norfair.

## Setup & Run

```bash
# setup
brew install rust go uv opencv

# run
# - generates deterministic data
# - runs tracking over deterministic data
bash run_benchmarks.sh
```

## Results

Benchmarks run on Apple M3 Pro. Results show significant variance between runs.

### Latest Results (v0.1.0 + optimizations)

| Scenario | Objects | Frames | Detections | Python FPS | Go FPS | Rust FPS | Rust vs Go |
|----------|---------|--------|------------|------------|--------|----------|------------|
| small    | 5       | 100    | 446        | ~4,800     | ~295k  | ~370k    | 1.2x       |
| medium   | 20      | 500    | 9,015      | ~540       | ~32k   | ~82k     | 2.6x       |
| large    | 50      | 1000   | 44,996     | ~103       | ~3.8k  | ~36k     | 9.5x       |
| stress   | 100     | 2000   | 179,789    | skipped    | ~550   | ~18k     | 33x        |

### Tracked Objects (Correctness Check)

All implementations produce matching tracked object counts:

| Scenario | Python  | Go      | Rust    |
|----------|---------|---------|---------|
| small    | 480     | 480     | 480     |
| medium   | 9,935   | 9,935   | 9,935   |
| large    | 49,788  | 49,788  | 49,788  |
| stress   | 199,602 | 199,602 | 199,602 |

---

## Optimization History

### v0.1.0 Baseline (Box<dyn Trait>)

Initial implementation using dynamic dispatch:

| Scenario | Rust FPS | Go FPS | Rust vs Go |
|----------|----------|--------|------------|
| small    | 477k     | 258k   | 1.9x       |
| medium   | 44k      | 32k    | 1.4x       |
| large    | 32k      | 3.9k   | 8.2x       |
| stress   | 17k      | 562    | 30x        |

### After Enum Dispatch (replace Box<dyn> with enum)

Replaced `Box<dyn Distance>`, `Box<dyn Filter>`, `Box<dyn FilterFactory>` with enum-based static dispatch:

| Scenario | Rust FPS | Go FPS | Rust vs Go |
|----------|----------|--------|------------|
| small    | 153k-449k | 262k-290k | 0.5-1.7x |
| medium   | 57k-114k  | 32k      | 1.8-3.5x |
| large    | 33k-42k   | 3.8k     | 8.7-11x  |
| stress   | 17k       | 550      | 31x      |

**Note:** High variance on small workload due to measurement noise at sub-millisecond timescales.

### After Vec<bool> Matching (replace HashSet with Vec<bool>)

Optimized greedy matching to use `Vec<bool>` instead of `HashSet<usize>`:

| Scenario | Rust FPS | Go FPS | Rust vs Go |
|----------|----------|--------|------------|
| small    | 200k-370k | 295k-304k | 0.7-1.2x |
| medium   | 68k-87k   | 31k-34k   | 2.0-2.8x |
| large    | 36k-40k   | 3.8k      | 9.5-10.5x |
| stress   | 18k-19k   | 549-551   | 33-35x   |

---

## Performance Notes

- **Small workloads:** High variance due to sub-millisecond timing. Rust and Go are comparable.
- **Medium to stress:** Rust scales significantly better than Go due to:
  - Enum-based static dispatch (no vtable overhead)
  - `Vec<bool>` for greedy matching (faster than HashSet)
  - Optimized Kalman filter implementation
  - Better memory locality with nalgebra matrices

