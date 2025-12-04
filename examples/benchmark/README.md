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

| Scenario | Objects | Frames | Detections | Python FPS | Go FPS | Rust FPS | Go vs Python |
|----------|---------|--------|------------|------------|--------|----------|--------------|
| small    | 5       | 100    | 446        | 4,825      | 298,322 | 19,012*  | 62x          |
| medium   | 20      | 500    | 9,015      | 545        | 34,031  | 2,482*   | 62x          |
| large    | 50      | 1000   | 44,996     | 103        | 3,779   | 329*     | 37x          |
| stress   | 100     | 2000   | 179,789    | 27         | 550     | 27*      | 20x          |

### Tracked Objects (Correctness Check)

| Scenario | Python | Go     | Rust  |
|----------|--------|--------|-------|
| small    | 480    | 480    | 6*    |
| medium   | 9,935  | 9,935  | 278*  |
| large    | 49,788 | 49,795 | 517*  |
| stress   | 199,602| 199,600| 1,819*|

**\*Note:** Rust implementation has a known tracking bug - objects are not being matched correctly. Investigation needed.
