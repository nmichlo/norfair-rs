# Norfair Rust Port - Implementation Plan

> **Reference:** See full architecture details in [~/.claude/plans/twinkly-strolling-goose.md](file:///Users/nmichlo/.claude/plans/twinkly-strolling-goose.md)

**Goal:** 100% equivalent Rust port of Python norfair, following the Go port structure closely while using Rust best practices.

**Key Principle:** The Rust port must be structurally equivalent to the Go port (`../norfair-go`), just in idiomatic Rust.

---

## Phase 1: Project Setup

- [x] Create `Cargo.toml` with dependencies (nalgebra, thiserror, approx)
- [x] Create `LICENSE` (BSD 3-Clause, matching Go port)
- [x] Create `THIRD_PARTY_LICENSES.md` (filterpy MIT, scipy BSD, motmetrics MIT)
- [x] Create `src/lib.rs` with module structure
- [x] Create `PROGRESS.md` for tracking implementation status
- [x] Copy test fixtures from Go port (`testdata/extended_metrics/*.txt`)
- [x] Copy golden images for drawing tests (`testdata/drawing/*.png`)

---

## Phase 2: Internal Dependencies (port from Go `internal/`)

### 2.1 Scipy Distance Functions (`internal/scipy/`)
- [x] Port `internal/scipy/distance.go` → `src/internal/scipy/distance.rs`
- [x] **Tests from Go:**
  - [x] `TestCdist_Euclidean`
  - [x] `TestCdist_Manhattan`
  - [x] `TestCdist_Cosine`
  - [x] `TestCdist_Chebyshev`
  - [ ] `TestCdist_SquaredEuclidean`

### 2.2 FilterPy Kalman Filter (`internal/filterpy/`)
- [x] Port `internal/filterpy/kalman.go` → `src/internal/filterpy/kalman.rs`
- [x] **Tests from Go:**
  - [x] `TestKalmanFilter_Create`
  - [x] `TestKalmanFilter_Predict`
  - [x] `TestKalmanFilter_Update`
  - [ ] `TestKalmanFilter_PredictUpdate`
  - [ ] `TestKalmanFilter_MultipleCycles`

### 2.3 NumPy Array Utilities (`internal/numpy/`)
- [x] Port `internal/numpy/array.go` → `src/internal/numpy/array.rs`
- [x] **Tests from Go:**
  - [x] `TestFlatten`
  - [x] `TestReshape`
  - [x] `TestValidatePoints`

### 2.4 MOT Metrics (`internal/motmetrics/`)
- [x] Port `internal/motmetrics/accumulator.go` → `src/internal/motmetrics/accumulator.rs`
- [x] Port `internal/motmetrics/iou.go` → `src/internal/motmetrics/iou.rs`
- [x] **Tests from Go:**
  - [x] `TestAccumulator_Update`
  - [x] `TestAccumulator_GetEvents`
  - [x] `TestIOUMatrix`
  - [x] `TestIOUMatrix_NoOverlap`
  - [x] `TestIOUMatrix_PartialOverlap`

---

## Phase 3: Filter Module (`pkg/norfairgo/filter*.go`)

### 3.1 Filter Traits
- [x] Create `src/filter/mod.rs`
- [x] Create `src/filter/traits.rs` with `Filter` and `FilterFactory` traits

### 3.2 OptimizedKalmanFilter
- [x] Port `optimized_kalman.go` → `src/filter/optimized.rs`
- [x] **Tests from Go (`filter_test.go`):**
  - [x] `TestOptimizedKalmanFilterFactory_Create`
  - [x] `TestOptimizedKalmanFilter_StaticObject`
  - [x] `TestOptimizedKalmanFilter_MovingObject`
  - [ ] `TestOptimizedKalmanFilter_PartialMeasurement`
- [ ] **Tests from Python (`test_tracker.py`):**
  - [ ] `test_simple` with OptimizedKalmanFilterFactory
  - [ ] `test_moving` with OptimizedKalmanFilterFactory
  - [ ] `test_distance_t` with OptimizedKalmanFilterFactory

### 3.3 FilterPyKalmanFilter
- [x] Port `filterpy_kalman.go` → `src/filter/filterpy.rs`
- [x] **Tests from Go:**
  - [x] `TestFilterPyKalmanFilterFactory_Create`
  - [x] `TestFilterPyKalmanFilter_StaticObject`
  - [x] `TestFilterPyKalmanFilter_MovingObject`
  - [ ] `TestFilterPyKalmanFilter_PartialMeasurement`
- [ ] **Tests from Python:**
  - [ ] `test_simple` with FilterPyKalmanFilterFactory
  - [ ] `test_moving` with FilterPyKalmanFilterFactory
  - [ ] `test_distance_t` with FilterPyKalmanFilterFactory

### 3.4 NoFilter
- [x] Port `no_filter.go` → `src/filter/no_filter.rs`
- [x] **Tests from Go:**
  - [x] `TestNoFilterFactory_Create`
  - [x] `TestNoFilter_Predict`
  - [x] `TestNoFilter_Update`

### 3.5 Filter Comparison Tests
- [ ] **Tests from Go:**
  - [ ] `TestFilterComparison_StaticObject`
  - [ ] `TestFilterComparison_MovingObject`
  - [ ] `TestFilters_MultiPoint`

---

## Phase 4: Distances Module (`pkg/norfairgo/distances*.go`)

### 4.1 Distance Traits
- [x] Create `src/distances/mod.rs`
- [x] Create `src/distances/traits.rs` with `Distance` trait
- [x] Port `ScalarDistance` wrapper
- [x] Port `VectorizedDistance` wrapper
- [x] Port `ScipyDistance` wrapper

### 4.2 Scalar Distance Functions
- [x] Port `Frobenius` function
- [x] Port `MeanManhattan` function
- [x] Port `MeanEuclidean` function
- [x] Port `CreateKeypointsVotingDistance` factory
- [x] Port `CreateNormalizedMeanEuclideanDistance` factory

### 4.3 Vectorized Distance Functions
- [x] Port `IoU` function (bounding box IoU)
- [ ] Port `IoUOpt` function (optimized version from Go)

### 4.4 Distance Registry
- [x] Port `GetDistanceByName` / `DistanceByName` function

### 4.5 Distance Tests

**From Python (`test_distances.py`):**
- [x] `test_frobenius` (partial - 2 cases: perfect_match, distance)
- [x] `test_mean_euclidean` (partial - 2 cases: perfect_match, distance)
- [x] `test_iou` (partial - 3 cases: perfect_match, no_overlap, partial_overlap)
- [ ] `test_mean_manhattan` (7 cases)
- [ ] `test_keypoint_vote` (7 cases)
- [ ] `test_normalized_euclidean` (9 cases)
- [ ] `test_scalar_distance` (wrapper test)
- [ ] `test_vectorized_distance` (wrapper test)
- [ ] `test_scipy_distance` (euclidean cdist test)

**From Go (`distances_test.go`):**
- [ ] `TestFrobenius` (7 cases)
- [ ] `TestMeanManhattan` (7 cases)
- [ ] `TestMeanEuclidean` (9 cases)
- [ ] `TestIoU` (6 cases)
- [ ] `TestIoU_InvalidBbox` (panic test)
- [ ] `TestScalarDistance` (wrapper)
- [ ] `TestVectorizedDistance` (wrapper)
- [ ] `TestScipyDistance` (euclidean)
- [ ] `TestKeypointVote` (7 cases)
- [ ] `TestNormalizedEuclidean` (9 cases)
- [ ] `TestGetDistanceByName` (frobenius, iou, euclidean, invalid)

---

## Phase 5: Core Tracker Module (`pkg/norfairgo/tracker*.go`)

### 5.1 Core Types
- [x] Create `src/detection.rs` with `Detection` struct
- [x] Create `src/tracked_object.rs` with `TrackedObject` struct
- [x] Create `src/tracker.rs` with `Tracker` struct and `TrackerConfig`
- [x] Port `TrackedObjectFactory` (ID generation)

### 5.2 Matching Algorithm
- [x] Port `matching.go` → `src/matching.rs`
- [x] Port greedy minimum-distance matching
- [x] **Tests from Go (`matching_test.go`):**
  - [x] `TestMatchDetectionsAndObjects_Empty`
  - [x] `TestMatchDetectionsAndObjects_SingleMatch`
  - [x] `TestMatchDetectionsAndObjects_MultipleMatches` (greedy)
  - [x] `TestMatchDetectionsAndObjects_ThresholdFiltering`
  - [x] `TestGetUnmatched`

### 5.3 Tracker Methods
- [x] Port `Tracker.Update()`
- [x] Port `TrackedObject.TrackerStep()` (inlined in update)
- [x] Port `TrackedObject.Hit()`
- [ ] Port `TrackedObject.Merge()` (ReID)
- [x] Port `TrackedObject.GetEstimate()`

### 5.4 Tracker Tests

**From Python (`test_tracker.py`):**
- [ ] `test_params` (invalid initializations)
- [ ] `test_simple` (parametrized: delay 0,1,3 × counter_max variations)
- [ ] `test_moving` (moving object, velocity estimation)
- [ ] `test_distance_t` (distance threshold behavior)
- [ ] `test_1d_points` (rank 1 detection handling)
- [ ] `test_camera_motion` (coordinate transformation)
- [ ] `test_count` (total_object_count, current_object_count)
- [ ] `test_multiple_trackers` (isolated tracker instances)
- [ ] `test_reid_hit_counter` (ReID matching and counters)

**From Go (`tracker_test.go`):**
- [x] `TestTracker_NewTracker`
- [x] `TestTracker_InvalidInitializationDelay`
- [x] `TestTracker_SimpleUpdate`
- [ ] `TestTracker_UpdateEmptyDetections`
- [x] `TestDetection_Creation`
- [ ] `TestTrackedObject_Creation`
- [ ] `TestTracker_CameraMotion` (1D and 2D points)

**From Go (`tracker_factory_test.go`):**
- [ ] `TestTrackedObjectFactory_CreateObject`
- [ ] `TestTrackedObjectFactory_IDAssignment`
- [ ] `TestTrackedObjectFactory_GlobalID`

---

## Phase 6: Camera Motion Module (`pkg/norfairgo/camera_motion*.go`)

### 6.1 Transformations
- [x] Create `src/camera_motion/mod.rs`
- [x] Create `src/camera_motion/transformations.rs`
- [x] Port `CoordinateTransformation` trait
- [x] Port `TranslationTransformation`
- [x] Port `NilCoordinateTransformation`
- [ ] Port `HomographyTransformation` (requires OpenCV feature)

### 6.2 Motion Estimator
- [x] Port `TransformationGetter` trait
- [x] Port `TranslationTransformationGetter`
- [ ] Port `HomographyTransformationGetter` (requires OpenCV feature)
- [ ] Port `MotionEstimator` (requires OpenCV feature)

### 6.3 Camera Motion Tests

**From Go (`camera_motion_test.go`):**
- [x] `TestNilTransformation`
- [x] `TestTranslationTransformation_AbsToRel`
- [x] `TestTranslationTransformation_RelToAbs`
- [x] `TestTranslationTransformation_Roundtrip`
- [x] `TestTranslationTransformationGetter`
- [ ] `TestHomographyTransformation_AbsToRel`
- [ ] `TestHomographyTransformation_RelToAbs`
- [ ] `TestNewHomographyTransformation`

---

## Phase 7: Metrics Module (`pkg/norfairgo/metrics*.go`)

### 7.1 Core Metrics
- [x] Create `src/metrics/mod.rs`
- [x] Port `InformationFile` (seqinfo.ini parser)
- [x] Port `PredictionsTextFile` (MOT format writer)
- [x] Port `DetectionFileParser` (MOT format reader)
- [x] Port `MOTAccumulator`
- [x] Port `MOTMetrics` / `MOTMetricsFromAccumulator`
- [ ] Port `EvalMotChallenge()` function (partial)

### 7.2 Metrics Tests

**From Go (`metrics_test.go`):**
- [x] `TestInformationFile_SearchInt`
- [x] `TestInformationFile_SearchString`
- [x] `TestInformationFile_SearchNotFound`
- [ ] `TestPredictionsTextFile_Write`
- [ ] `TestDetectionFileParser_Parse`

**From Rust:**
- [x] `TestAccumulator_Empty`
- [x] `TestAccumulator_PerfectTracking`
- [x] `TestAccumulator_AllMisses`
- [x] `TestMetricsFromAccumulator`

**From Go (`extended_metrics_test.go`):**
- [ ] `TestEvalMotChallenge_Perfect` (MOTA ≈ 1.0, no FP/misses/switches)
- [ ] `TestEvalMotChallenge_MostlyLost` (low MOTA, many misses)
- [ ] `TestEvalMotChallenge_Fragmented` (ID switches/fragmentations)
- [ ] `TestEvalMotChallenge_Mixed` (realistic scenario, valid ranges)

---

## Phase 8: Utils Module

- [x] Create `src/utils.rs`
- [x] Port `validate_points`
- [x] Port `warn_once`
- [x] Port `any_true` / `all_true`
- [x] Port `get_bounding_box`
- [x] Port `clamp`
- [ ] Port `GetCutout`
- [ ] Port `PrintObjectsAsTable`

**From Go (`utils_test.go`):**
- [x] `TestValidatePoints_Valid`
- [x] `TestValidatePoints_Invalid`
- [x] `TestAnyTrue`
- [x] `TestAllTrue`
- [x] `TestGetBoundingBox`
- [x] `TestClamp`

---

## Phase 9: Video Module (requires OpenCV feature)

- [ ] Create `src/video.rs` with `#[cfg(feature = "opencv")]`
- [ ] Port `Video` struct (VideoCapture wrapper)
- [ ] Port iterator implementation

**From Go (`video_test.go`):**
- [ ] `TestVideo_OpenFile`
- [ ] `TestVideo_GetFrame`
- [ ] `TestVideo_Write`

---

## Phase 10: Drawing Module (LOWEST PRIORITY, requires OpenCV)

### 10.1 Core Drawing
- [ ] Create `src/drawing/mod.rs`
- [ ] Port `Drawer` (low-level primitives)
- [ ] Port `Drawable` trait

**From Go (`drawer_test.go`):**
- [ ] `TestDrawer_Circle`
- [ ] `TestDrawer_Rectangle`
- [ ] `TestDrawer_Line`
- [ ] `TestDrawer_Text`
- [ ] `TestDrawer_Polygon`

### 10.2 Color System
- [ ] Port `Color` struct and constants (140+ named colors)
- [ ] Port `Palette` (by_id, by_label, random strategies)
- [ ] Port `HexToBGR`

**From Go (`color_test.go` in both packages):**
- [ ] `TestColor_Constants`
- [ ] `TestHexToBGR`
- [ ] `TestPalette_ByID`
- [ ] `TestPalette_ByLabel`

### 10.3 Path Drawing
- [ ] Port `Paths` / `AbsolutePaths`

**From Go (`path_test.go`):**
- [ ] `TestPaths_Add`
- [ ] `TestPaths_Draw`
- [ ] `TestAbsolutePaths`

### 10.4 Draw Functions
- [ ] Port `DrawPoints` / `DrawTrackedObjects`
- [ ] Port `DrawBoxes` / `DrawTrackedBoxes`
- [ ] Port `DrawAbsoluteGrid`

**From Go (`draw_points_test.go`, `draw_boxes_test.go`, `absolute_grid_test.go`):**
- [ ] `TestDrawPoints_Basic`
- [ ] `TestDrawPoints_DirectColor` (golden image comparison)
- [ ] `TestDrawBoxes_Basic`
- [ ] `TestDrawBoxes_DirectColor` (golden image comparison)
- [ ] `TestDrawAbsoluteGrid`

### 10.5 Fixed Camera
- [ ] Port `FixedCamera` context

**From Go (`fixed_camera_test.go`):**
- [ ] `TestFixedCamera_Create`
- [ ] `TestFixedCamera_AdjustFrame`

---

## Phase 11: Integration Tests

**From Go (`integration_test.go`):**
- [ ] `TestIntegration_CompleteTrackingPipeline` (20 frames, 2 objects)
- [ ] `TestIntegration_MultipleFilterTypes` (OptimizedKalman, FilterPy, NoFilter)
- [ ] `TestIntegration_MultipleDistanceFunctions` (IoU, Euclidean, CustomScalar)
- [ ] `TestIntegration_ReIDEnabled` (occlusion and recovery)
- [ ] `TestIntegration_CameraMotionCompensation` (translation transform)
- [ ] `TestIntegration_EndToEndWorkflow` (tracking + drawing)

---

## Phase 12: Benchmarks

**From Go (`benchmark_test.go`):**
- [ ] `BenchmarkTracker_Update`
- [ ] `BenchmarkIoU`
- [ ] `BenchmarkFrobenius`
- [ ] `BenchmarkOptimizedKalmanFilter_Predict`
- [ ] `BenchmarkOptimizedKalmanFilter_Update`

---

## Phase 13: PyO3 Python Bindings (LAST)

- [ ] Add `pyo3` and `numpy` dependencies (optional feature)
- [ ] Create `pyproject.toml` for Maturin
- [ ] Create Python wrapper module
- [ ] Implement `Detection` class binding
- [ ] Implement `Tracker` class binding
- [ ] Implement `TrackedObject` class binding
- [ ] Implement distance function bindings
- [ ] Implement filter factory bindings
- [ ] Create Python type stubs (`.pyi` files)
- [ ] Test drop-in replacement compatibility with original norfair

---

## Test Data Files to Copy

From `../norfair-go/testdata/`:
```
testdata/
├── extended_metrics/
│   ├── gt1.txt      # Perfect tracking ground truth
│   ├── gt2.txt      # Mostly lost ground truth
│   ├── gt3.txt      # Fragmented ground truth
│   ├── gt4.txt      # Mixed scenario ground truth
│   ├── pred1.txt    # Perfect predictions
│   ├── pred2.txt    # Mostly lost predictions
│   ├── pred3.txt    # Fragmented predictions
│   └── pred4.txt    # Mixed scenario predictions
└── drawing/
    ├── draw_points_direct_color_golden.png
    ├── draw_boxes_direct_color_golden.png
    └── drawing_primitives_golden.png
```

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

See full comparison in [~/.claude/plans/twinkly-strolling-goose.md](file:///Users/nmichlo/.claude/plans/twinkly-strolling-goose.md#numerical-library-comparison)
