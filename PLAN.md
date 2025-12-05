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

**Tests:** 285 total (277 unit + 2 fixture + 6 integration) - All passing ✅

### Performance Optimization Complete (2025-12-04)

Phase 1 enum dispatch optimization achieved 2.78x improvement on small workloads:

| Scenario | Before | After | Improvement | vs Go |
|----------|--------|-------|-------------|-------|
| Small | 148,469 fps | 413,081 fps | **2.78x** | **1.46x faster** |
| Medium | 63,497 fps | 93,049 fps | **1.47x** | **2.89x faster** |
| Large | 32,778 fps | 38,621 fps | **1.18x** | faster |
| Stress | 16,510 fps | 18,169 fps | **1.10x** | faster |

**Key changes:**
- Replaced `Box<dyn Distance>` with `DistanceFunction` enum
- Replaced `Box<dyn Filter>` with `FilterEnum` enum
- Replaced `Box<dyn FilterFactory>` with `FilterFactoryEnum` enum
- All dispatch now uses static enum matching instead of vtable lookups

SEE: ./PLAN_TESTS.md for detailed checklist

### Fixture Tests Complete (2025-12-04)

E2E fixture tests verify Rust tracker output matches Python reference:
- `tests/fixture_tests.rs` - 2 tests (small, medium scenarios)
- Fixtures generated from Python norfair in `tests/data/fixtures/`
- Tolerance: 1e-6 for numerical comparisons

**Bugs fixed during fixture test implementation:**
1. **Hit counter decrement** - Was only decrementing for non-initializing objects; now decrements for ALL objects
2. **Hit counter increment** - Was `+period`; fixed to `+2*period` on match (net +1 with decay)
3. **Initialization check** - Was `>=`; fixed to `>` for `hit_counter > initialization_delay`
4. **Matrix flattening** - Was column-major; fixed to row-major to match Python/Go
5. **Kalman covariance** - Was updating in predict(); fixed to embed in update() like Go
6. **Object cleanup order** - Was decrement-then-remove; fixed to remove-then-decrement
7. **initializing_id persistence** - Was clearing on initialization; now keeps permanently

### Verification Complete (2025-12-04)

All non-OpenCV tests have been verified as ported:
- `internal/scipy/distance_test.go` - 13/13 ✅
- `internal/scipy/optimize_test.go` - 11/12 ✅ (1 N/A - Go-specific helper)
- `internal/filterpy/kalman_test.go` - 8/8 ✅
- `internal/motmetrics/iou_test.go` - 13/13 ✅
- `internal/motmetrics/accumulator_test.go` - 17/17 ✅
- `internal/numpy/array_test.go` - 14/14 ✅
- `pkg/norfairgo/matching_test.go` - 14/14 ✅
- `pkg/norfairgo/utils_test.go` - 10/16 ✅ (6 require OpenCV GetCutout)
- `pkg/norfairgo/camera_motion_test.go` - 9/26 ✅ (17 require OpenCV)

Deferred (OpenCV required): Video (18), Homography/MotionEstimator (17), Drawing/Color (111), GetCutout (6)

### Known Issues

1. **Concurrent test flakiness** - `test_factory_concurrent_multiple_factories` can fail when run in parallel due to global ID counter interference with other tests. Passes when run single-threaded (`--test-threads=1`).

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
- [x] Copy test fixtures from Go port (`tests/data/extended_metrics/*.txt`)
- [x] Copy golden images for drawing tests (`tests/data/drawing/*.png`)

---

## Phase 2: Internal Dependencies - ✅ COMPLETE

### 2.1 Scipy Distance Functions (`internal/scipy/`)
- [x] Port `internal/scipy/distance.go` → `src/internal/scipy/distance.rs`
- [x] Tests: ALL 13 tests ported (Euclidean, Manhattan, Cosine, Chebyshev, SquaredEuclidean, etc.)

### 2.2 Scipy Optimize Functions (`internal/scipy/`)
- [x] Port `internal/scipy/optimize.go` → `src/internal/scipy/optimize.rs`
- [x] Tests: ALL 11 relevant tests ported (LinearSumAssignment variants)

### 2.3 FilterPy Kalman Filter (`internal/filterpy/`)
- [x] Port `internal/filterpy/kalman.go` → `src/internal/filterpy/kalman.rs`
- [x] Tests: ALL 8 tests ported (Create, Predict, Update, Cycle, Partial, Singular, Getters, MultiDim)

### 2.4 NumPy Array Utilities (`internal/numpy/`)
- [x] Port `internal/numpy/array.go` → `src/internal/numpy/array.rs`
- [x] Tests: ALL 14 tests ported (Linspace variants, Flatten, Reshape, ValidatePoints)

### 2.5 MOT Metrics (`internal/motmetrics/`)
- [x] Port accumulator and IoU
- [x] Tests: ALL 13 IoU tests ported

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
- [x] 49 distance function tests passing
- [x] 8 wrapper tests passing (ScalarDistance, VectorizedDistance, ScipyDistance, distance_by_name)

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
- [x] 11 tracker tests passing
- [x] Ported: params, simple, moving, distance_t, 1d_points, count, reid, multiple_trackers

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

## Phase 7: Metrics Module - ✅ COMPLETE

### 7.1 Core Metrics
- [x] `InformationFile`, `PredictionsTextFile`, `DetectionFileParser`
- [x] `MOTAccumulator`, `MOTMetrics`
- [x] Extended metrics scenarios (via accumulator tests)

### 7.2 Metrics Tests
- [x] 47 accumulator tests passing
- [x] Extended metrics tests (Perfect, MostlyLost, Fragmented, Mixed) ported as accumulator tests

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

## Phase 12: Benchmarks - ✅ COMPLETE

- [x] Cross-language benchmark infrastructure created
- [x] Criterion benchmarks for Rust (6 benchmarks in benches/tracker_benchmarks.rs)

---

## Phase 13: PyO3 Python Bindings - ❌ NOT STARTED

- [ ] pyo3/numpy dependencies
- [ ] Python wrapper classes
- [ ] Drop-in replacement API

---

## Test Inventory

See [PLAN_TESTS.md](./PLAN_TESTS.md) for complete test porting checklist.

| Category | Tests |
|----------|-------|
| Filter | 15 |
| Distance (functions) | 49 |
| Distance (wrappers) | 8 |
| Tracker | 11 |
| TrackedObject & Factory | 12 |
| Detection | 4 |
| Matching | 16 |
| Camera motion | 4 |
| Metrics (accumulator) | 47 |
| Utils | 6 |
| Internal (filterpy, scipy, numpy, motmetrics) | 98 |
| Integration | 6 |
| Fixture (E2E) | 2 |
| **Total** | **278** |

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
- ALL objects (including initializing) decay hit_counter by 1 per frame, matched objects gain +2*period
- Matrix operations use row-major order to match Python/Go (via `flatten_row_major()` helper)
- Test fixtures from Go port needed for extended metrics tests
- Fixture tests verify exact numerical equivalence with Python reference implementation
