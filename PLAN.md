# Norfair Rust Port - Plan & Progress

> **Reference:** See full architecture details in [~/.claude/plans/twinkly-strolling-goose.md](file:///Users/nmichlo/.claude/plans/twinkly-strolling-goose.md)

**Goal:** 100% equivalent Rust port of Python norfair, following the Go port structure closely while using Rust best practices.

**Key Principle:** The Rust port must be structurally equivalent to the Go port (`../norfair-go`), just in idiomatic Rust.

---

## THE GOLDEN RULE

**When porting or tests fail showing numerical differences between python/go and Rust:**

**DO THIS FIRST:**
1. Find the Rust code that is failing
2. Find the correspinding python and go code
3. Open them all side-by-side
4. Compare them all line-by-line
5. Look for obvious bugs:
   - Wrong formulas (det(c*A) vs c^n*det(A))
   - Scalar vs array parameters
   - Missing loops or wrong loop bounds
   - Transposed matrices or wrong indexing
   - Incorrect logic or control flow
6. Add a test case for Rust that checks this divergence point.

**ONLY IF THAT FAILS (rare):**
1. Create minimal debug fixture in python
2. Add targeted debug output to Rust
3. Compare intermediate values
4. Trace divergence point
5. Add a test case for Rust that checks this divergence point.

## Current Status

**Tests:**

SEE: ./PLAN_TESTS.md

### Known Bugs

1. **Rust tracking bug** - Objects not matching correctly in benchmarks
   - IoU expects 1 row × 4 cols, benchmark uses 2 rows × 2 cols
   - Need to investigate VectorizedDistance wrapper

2. **Go minor differences** - Some tracking differences with python under stress test, need to investigate.

---

## Useful Commands

```bash
cargo check           # Check compilation
cargo test --release  # Run tests with release optimizations, faster.
```

---

## Phase 1: Project Setup - ✅ COMPLETE

- [x] Create `Cargo.toml` with dependencies (nalgebra, thiserror, approx)
- [x] Create `LICENSE` (BSD 3-Clause, matching Go port)
- [x] Create `THIRD_PARTY_LICENSES.md` (filterpy MIT, scipy BSD, motmetrics MIT)
- [x] Create `src/lib.rs` with module structure
- [x] Copy test fixtures from Go port (`testdata/extended_metrics/*.txt`)
- [x] Copy golden images for drawing tests (`testdata/drawing/*.png`)

---

## Phase 2: Internal Dependencies - ✅ COMPLETE

### 2.1 Scipy Distance Functions (`internal/scipy/`)
- [x] Port `internal/scipy/distance.go` → `src/internal/scipy/distance.rs`
- [x] Tests: Euclidean, Manhattan, Cosine, Chebyshev
- [ ] `TestCdist_SquaredEuclidean`

### 2.2 FilterPy Kalman Filter (`internal/filterpy/`)
- [x] Port `internal/filterpy/kalman.go` → `src/internal/filterpy/kalman.rs`
- [x] Tests: Create, Predict, Update
- [ ] `TestKalmanFilter_PredictUpdate`, `TestKalmanFilter_MultipleCycles`

### 2.3 NumPy Array Utilities (`internal/numpy/`)
- [x] Port `internal/numpy/array.go` → `src/internal/numpy/array.rs`
- [x] Tests: Flatten, Reshape, ValidatePoints

### 2.4 MOT Metrics (`internal/motmetrics/`)
- [x] Port accumulator and IoU
- [x] Tests: Accumulator_Update, GetEvents, IOUMatrix

---

## Phase 3: Filter Module - ✅ COMPLETE

### 3.1 Filter Traits
- [x] `src/filter/traits.rs` with `Filter` and `FilterFactory` traits

### 3.2 OptimizedKalmanFilter
- [x] Port `optimized_kalman.go` → `src/filter/optimized.rs`
- [x] Tests: Create, StaticObject, MovingObject
- [ ] PartialMeasurement test

### 3.3 FilterPyKalmanFilter
- [x] Port `filterpy_kalman.go` → `src/filter/filterpy.rs`
- [x] Tests: Create, StaticObject, MovingObject
- [ ] PartialMeasurement test

### 3.4 NoFilter
- [x] Port `no_filter.go` → `src/filter/no_filter.rs`
- [x] Tests: Create, Predict, Update

### 3.5 Filter Comparison Tests
- [ ] StaticObject, MovingObject, MultiPoint comparisons

---

## Phase 4: Distances Module - ✅ COMPLETE

### 4.1 Distance Traits & Wrappers
- [x] `Distance` trait, `ScalarDistance`, `VectorizedDistance`, `ScipyDistance`

### 4.2 Scalar Distance Functions
- [x] Frobenius, MeanManhattan, MeanEuclidean
- [x] KeypointsVotingDistance, NormalizedMeanEuclideanDistance

### 4.3 Vectorized Distance Functions
- [x] IoU (bounding box)
- [ ] IoUOpt (optimized version)

### 4.4 Distance Registry
- [x] `distance_by_name` function

### 4.5 Distance Tests
- [x] 49 distance tests passing
- [ ] Missing: wrapper tests, GetDistanceByName edge cases

---

## Phase 5: Core Tracker Module - ✅ COMPLETE

### 5.1 Core Types
- [x] `Detection`, `TrackedObject`, `Tracker`, `TrackerConfig`
- [x] `TrackedObjectFactory` (ID generation)

### 5.2 Matching Algorithm
- [x] Greedy minimum-distance matching
- [x] 16 matching tests passing

### 5.3 Tracker Methods
- [x] `Update()`, `TrackerStep()`, `Hit()`, `GetEstimate()`
- [ ] `Merge()` (ReID)

### 5.4 Tracker Tests
- [x] 4 tracker tests passing
- [ ] Missing: params, simple (parametrized), moving, distance_t, 1d_points, count, reid

---

## Phase 6: Camera Motion Module - ⚠️ PARTIAL

### 6.1 Transformations
- [x] `CoordinateTransformation` trait
- [x] `TranslationTransformation`, `NilCoordinateTransformation`
- [ ] `HomographyTransformation` (requires OpenCV)

### 6.2 Motion Estimator
- [x] `TransformationGetter` trait
- [x] `TranslationTransformationGetter`
- [ ] `HomographyTransformationGetter`, `MotionEstimator` (require OpenCV)

### 6.3 Camera Motion Tests
- [x] 4 tests passing
- [ ] Homography tests (require OpenCV)

---

## Phase 7: Metrics Module - ⚠️ PARTIAL

### 7.1 Core Metrics
- [x] `InformationFile`, `PredictionsTextFile`, `DetectionFileParser`
- [x] `MOTAccumulator`, `MOTMetrics`
- [ ] `EvalMotChallenge()` (partial)

### 7.2 Metrics Tests
- [x] 7 tests passing
- [ ] Extended metrics tests (Perfect, MostlyLost, Fragmented, Mixed)

---

## Phase 8: Utils Module - ✅ COMPLETE

- [x] validate_points, warn_once, any_true/all_true, get_bounding_box, clamp
- [x] 6 tests passing
- [ ] GetCutout, PrintObjectsAsTable

---

## Phase 9: Video Module - ❌ NOT STARTED

- [ ] `Video` struct with `#[cfg(feature = "opencv")]`

---

## Phase 10: Drawing Module - ❌ NOT STARTED

- [ ] Drawer, Color constants, Palette, Paths
- [ ] draw_points, draw_boxes, DrawAbsoluteGrid
- [ ] FixedCamera

---

## Phase 11: Integration Tests - ✅ COMPLETE

- [x] 6 integration tests passing
- [x] CompleteTrackingPipeline, MultipleFilterTypes, MultipleDistanceFunctions
- [x] ReIDEnabled, CameraMotionCompensation, ObjectLifecycle

---

## Phase 12: Benchmarks

- [x] Cross-language benchmark infrastructure created
- [ ] Criterion benchmarks for Rust

---

## Phase 13: PyO3 Python Bindings - ❌ NOT STARTED

- [ ] pyo3/numpy dependencies
- [ ] Python wrapper classes
- [ ] Drop-in replacement API

---

## Test Inventory

See [TODO_TESTS.md](./TODO_TESTS.md) for complete test porting checklist.

| Category | Tests |
|----------|-------|
| Filter | 15 |
| Distance | 49 |
| Tracker | 4 |
| TrackedObject & Factory | 12 |
| Detection | 4 |
| Matching | 16 |
| Camera motion | 4 |
| Metrics | 7 |
| Utils | 6 |
| Internal (filterpy, scipy, numpy, motmetrics) | 63 |
| Integration | 6 |
| **Total** | **186** |

---

## File Mapping: Go → Rust

| Go Source | Rust Target |
|-----------|-------------|
| `internal/scipy/distance.go` | `src/internal/scipy/distance.rs` |
| `internal/filterpy/kalman.go` | `src/internal/filterpy/kalman.rs` |
| `internal/numpy/array.go` | `src/internal/numpy/array.rs` |
| `internal/motmetrics/*.go` | `src/internal/motmetrics/*.rs` |
| `pkg/norfairgo/tracker.go` | `src/tracker.rs` |
| `pkg/norfairgo/detection.go` | `src/detection.rs` |
| `pkg/norfairgo/tracked_object.go` | `src/tracked_object.rs` |
| `pkg/norfairgo/distances.go` | `src/distances/*.rs` |
| `pkg/norfairgo/filter*.go` | `src/filter/*.rs` |
| `pkg/norfairgo/camera_motion.go` | `src/camera_motion/*.rs` |
| `pkg/norfairgo/metrics.go` | `src/metrics/*.rs` |
| `pkg/norfairgo/utils.go` | `src/utils.rs` |
| `pkg/norfairgo/video.go` | `src/video.rs` |
| `pkg/norfairgodraw/*.go` | `src/drawing/*.rs` |
| `pkg/norfairgocolor/*.go` | `src/drawing/color.rs` |

---

## Numerical Library Decision: **nalgebra** ✓

**Chosen:** `nalgebra` for the following reasons:
- Pure Rust (no external BLAS/LAPACK dependencies)
- Simple build process across all platforms
- Strong compile-time dimension checking where beneficial
- Good performance for small-to-medium matrices (typical in tracking)
- `DMatrix<f64>` and `DVector<f64>` for dynamic dimensions

---

## Notes

- Using nalgebra instead of ndarray (pure Rust, no BLAS required)
- Initializing objects don't decay hit_counter to allow accumulation
- Test fixtures from Go port needed for extended metrics tests
