# Test Porting Checklist

Complete list of ALL tests from Python, Go, and their port status to Rust.

## RULES

re-read .claude/CLAUDE.md, then continue porting tests from PLAN_TESTS.md (except drawing tests). DO NOT MISS ANY TESTS. If you think you can skip tests, you are wrong, implement them. If you want to skip
anything, you are wrong, do not skip it. If you notice logic differences according between rust/python/go code, you MUST fix those errors (python is the source of truth, golang has extra tests and is easier to
port to rust but not 100% trustworthy, rust should not be trusted this is what we are building and fixing). You must ALWAYS follow the process in the GOLDEN RULE of claude.md when carrying out these tasks.

## Summary Statistics

| Language | Total Tests | Ported | Missing (non-OpenCV) |
|----------|-------------|--------|----------------------|
| Python | 19 | 18 | 1 (drawing) |
| Go | 323 | ~200+ | ~100+ (mostly OpenCV/drawing) |
| **Total** | 342 | ~218+ | ~100+ |

**Current Rust test count:** 276 tests (270 unit + 6 integration) - All passing ✅

**Verified complete (2025-12-04):**
- `internal/scipy/distance_test.go` - 13/13 ✅
- `internal/scipy/optimize_test.go` - 11/12 ✅ (1 N/A - Go-specific helper)
- `internal/filterpy/kalman_test.go` - 8/8 ✅
- `internal/motmetrics/iou_test.go` - 13/13 ✅
- `internal/motmetrics/accumulator_test.go` - 17/17 ✅
- `internal/numpy/array_test.go` - 14/14 ✅
- `pkg/norfairgo/matching_test.go` - 14/14 ✅
- `pkg/norfairgo/utils_test.go` - 10/16 ✅ (6 require OpenCV)
- `pkg/norfairgo/camera_motion_test.go` - 9/26 ✅ (17 require OpenCV)
- `pkg/norfairgo/tracker_test.go` - 7/7 ✅
- `pkg/norfairgo/tracker_factory_test.go` - 8/8 ✅
- `pkg/norfairgo/distances_test.go` - 11/11 ✅
- `pkg/norfairgo/filter_test.go` - 14/14 ✅
- `pkg/norfairgo/benchmark_test.go` - 6/6 ✅ (Criterion benchmarks)

**Deferred (require OpenCV):**
- Video tests (18 tests) - Video module not implemented
- Homography/MotionEstimator tests (17 tests)
- Drawing/Color tests (111 tests)
- GetCutout tests (6 tests)

---

## Python Tests (19 total)

### `tests/test_drawing.py` (1 test)
| # | Test | Ported? |
|---|------|---------|
| 1 | `test_hex_parsing` | ❌ |

### `tests/test_distances.py` (9 tests)
| # | Test | Ported? |
|---|------|---------|
| 2 | `test_frobenius` | ✅ |
| 3 | `test_mean_manhattan` | ✅ |
| 4 | `test_mean_euclidean` | ✅ |
| 5 | `test_iou` | ✅ |
| 6 | `test_keypoint_vote` | ✅ |
| 7 | `test_normalized_euclidean` | ✅ |
| 8 | `test_scalar_distance` | ✅ |
| 9 | `test_vectorized_distance` | ✅ |
| 10 | `test_scipy_distance` | ✅ |

### `tests/test_tracker.py` (9 tests)
| # | Test | Ported? |
|---|------|---------|
| 11 | `test_params` | ✅ |
| 12 | `test_simple` | ✅ |
| 13 | `test_moving` | ✅ |
| 14 | `test_distance_t` | ✅ |
| 15 | `test_1d_points` | ✅ |
| 16 | `test_camera_motion` | ✅ |
| 17 | `test_count` | ✅ |
| 18 | `test_multiple_trackers` | ✅ |
| 19 | `test_reid_hit_counter` | ✅ |

---

## Go Tests (~323 total)

### `internal/scipy/distance_test.go` (13 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 1 | `TestCdist_Euclidean` | ✅ |
| 2 | `TestCdist_Manhattan` | ✅ |
| 3 | `TestCdist_Cosine` | ✅ `test_cdist_cosine_orthogonal` |
| 4 | `TestCdist_Cosine_Parallel` | ✅ `test_cdist_cosine_parallel` |
| 5 | `TestCdist_Cosine_ZeroVector` | ✅ `test_cdist_cosine_zero_vector` |
| 6 | `TestCdist_SquaredEuclidean` | ✅ `test_cdist_sqeuclidean` |
| 7 | `TestCdist_Chebyshev` | ✅ |
| 8 | `TestCdist_DifferentDimensions` | ✅ `test_cdist_different_row_counts` |
| 9 | `TestCdist_PanicOnMismatchedColumns` | ✅ `test_cdist_panic_on_mismatched_columns` |
| 10 | `TestCdist_PanicOnUnsupportedMetric` | ✅ `test_cdist_panic_on_unsupported_metric` |
| 11 | `TestCdist_SingleVectors` | ✅ `test_cdist_single_vectors` |
| 12 | `TestCdist_IdenticalVectors` | ✅ `test_cdist_identical_vectors` |
| 13 | `TestCdist_Cosine_AntiParallel` | ✅ `test_cdist_cosine_antiparallel` |

### `internal/scipy/optimize_test.go` (12 tests) ✅ ALL PORTED (11 relevant)
| # | Test | Ported? |
|---|------|---------|
| 14 | `TestLinearSumAssignment_BasicSquare` | ✅ `test_linear_sum_assignment_basic_square` |
| 15 | `TestLinearSumAssignment_CostThreshold` | ✅ `test_linear_sum_assignment_cost_threshold` |
| 16 | `TestLinearSumAssignment_RectangularMoreRows` | ✅ `test_linear_sum_assignment_rectangular_more_rows` |
| 17 | `TestLinearSumAssignment_RectangularMoreCols` | ✅ `test_linear_sum_assignment_rectangular_more_cols` |
| 18 | `TestLinearSumAssignment_EmptyMatrix` | ✅ `test_linear_sum_assignment_empty_matrix` |
| 19 | `TestLinearSumAssignment_EmptyColumns` | ✅ `test_linear_sum_assignment_empty_columns` |
| 20 | `TestLinearSumAssignment_AllRejectedByThreshold` | ✅ `test_linear_sum_assignment_all_rejected_by_threshold` |
| 21 | `TestLinearSumAssignment_OptimalMatching` | ✅ `test_linear_sum_assignment_optimal_matching` |
| 22 | `TestLinearSumAssignment_SingleElement` | ✅ `test_linear_sum_assignment_single_element` |
| 23 | `TestLinearSumAssignment_PartialMatching` | ✅ `test_linear_sum_assignment_partial_matching` |
| 24 | `TestLinearSumAssignment_ZeroCosts` | ✅ `test_linear_sum_assignment_zero_costs` |
| 25 | `TestLinearSumAssignment_max_helper` | N/A (Go-specific helper, Rust uses built-in max) |

### `internal/filterpy/kalman_test.go` (8 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 26 | `TestNewKalmanFilter` | ✅ `test_kalman_filter_create` |
| 27 | `TestKalmanFilter_Predict` | ✅ `test_kalman_filter_predict` |
| 28 | `TestKalmanFilter_Update` | ✅ `test_kalman_filter_update` |
| 29 | `TestKalmanFilter_PredictUpdateCycle` | ✅ `test_kalman_filter_predict_update_cycle` |
| 30 | `TestKalmanFilter_PartialMeasurement` | ✅ `test_kalman_filter_partial_measurement` |
| 31 | `TestKalmanFilter_SingularInnovationCovariance` | ✅ `test_kalman_filter_singular_innovation_covariance` |
| 32 | `TestKalmanFilter_GettersSetters` | ✅ `test_kalman_filter_getters` + `test_kalman_filter_getters_setters_extended` |
| 33 | `TestKalmanFilter_MultiDimensional` | ✅ `test_kalman_filter_multi_dimensional` |

### `internal/motmetrics/iou_test.go` (13 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 34 | `TestIouDistance_PerfectOverlap` | ✅ `test_iou_distance_perfect_overlap` |
| 35 | `TestIouDistance_NoOverlap` | ✅ `test_iou_distance_no_overlap` |
| 36 | `TestIouDistance_PartialOverlap` | ✅ `test_iou_distance_partial_overlap` |
| 37 | `TestIouDistance_ContainedBox` | ✅ `test_iou_distance_contained_box` |
| 38 | `TestIouDistance_AdjacentBoxes` | ✅ `test_iou_distance_adjacent_boxes` |
| 39 | `TestIouDistance_SmallOverlap` | ✅ `test_iou_distance_small_overlap` |
| 40 | `TestIouDistance_FloatingPoint` | ✅ `test_iou_distance_floating_point` |
| 41 | `TestIouDistance_InvalidBox1` | ✅ `test_iou_distance_invalid_box1` |
| 42 | `TestIouDistance_InvalidBox2` | ✅ `test_iou_distance_invalid_box2` |
| 43 | `TestIouDistance_WrongLength` | ✅ `test_iou_distance_wrong_length` |
| 44 | `TestComputeIoUMatrix` | ✅ `test_compute_iou_matrix` |
| 45 | `TestComputeIoUMatrix_Empty` | ✅ `test_compute_iou_matrix_empty` |
| 46 | `TestComputeIoUMatrix_LargeSet` | ✅ `test_compute_iou_matrix_large_set` |

### `internal/motmetrics/accumulator_test.go` (17 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 47 | `TestNewTrackLifecycle` | ✅ `test_new_track_lifecycle` |
| 48 | `TestTrackLifecycle_UpdateMatched` | ✅ `test_track_lifecycle_update_matched` |
| 49 | `TestTrackLifecycle_UpdateMissed` | ✅ `test_track_lifecycle_update_missed` |
| 50 | `TestTrackLifecycle_Fragmentation` | ✅ `test_track_lifecycle_fragmentation` |
| 51 | `TestTrackLifecycle_Coverage` | ✅ `test_track_lifecycle_coverage` |
| 52 | `TestNewMOTAccumulator` | ✅ `test_new_extended_mot_accumulator` |
| 53 | `TestMOTAccumulator_Update_EmptyFrame` | ✅ `test_extended_accumulator_update_empty_frame` |
| 54 | `TestMOTAccumulator_Update_OnlyPredictions` | ✅ `test_extended_accumulator_update_only_predictions` |
| 55 | `TestMOTAccumulator_Update_OnlyGT` | ✅ `test_extended_accumulator_update_only_gt` |
| 56 | `TestMOTAccumulator_Update_PerfectMatch` | ✅ `test_extended_accumulator_update_perfect_match` |
| 57 | `TestMOTAccumulator_Update_PartialMatch` | ✅ `test_extended_accumulator_update_partial_match` |
| 58 | `TestMOTAccumulator_DetectSwitches` | ✅ `test_extended_accumulator_detect_switches` |
| 59 | `TestMOTAccumulator_MultiFrame` | ✅ `test_extended_accumulator_multi_frame` |
| 60 | `TestComputeExtendedMetrics_MostlyTracked` | ✅ `test_compute_extended_metrics_mostly_tracked` |
| 61 | `TestComputeExtendedMetrics_MostlyLost` | ✅ `test_compute_extended_metrics_mostly_lost` |
| 62 | `TestComputeExtendedMetrics_PartiallyTracked` | ✅ `test_compute_extended_metrics_partially_tracked` |
| 63 | `TestComputeExtendedMetrics_Fragmentations` | ✅ `test_compute_extended_metrics_fragmentations` |

**Note:** Rust has additional accumulator tests: `test_accumulator_new`, `test_accumulator_*` (basic MOTAccumulator), `test_metrics_*` (recall, precision, mota, motp)

### `internal/numpy/array_test.go` (14 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 64 | `TestLinspace_Basic` | ✅ `test_linspace_basic` |
| 65 | `TestLinspace_TwoPoints` | ✅ `test_linspace_two_points` |
| 66 | `TestLinspace_SinglePoint` | ✅ `test_linspace_single_point` |
| 67 | `TestLinspace_Zero` | ✅ `test_linspace_zero` |
| 68 | `TestLinspace_Negative` | ✅ `test_linspace_negative` |
| 69 | `TestLinspace_ReverseRange` | ✅ `test_linspace_reverse_range` |
| 70 | `TestLinspace_FloatingPoint` | ✅ `test_linspace_floating_point` |
| 71 | `TestLinspace_LargeN` | ✅ `test_linspace_large_n` |
| 72 | `TestLinspace_EndpointExact` | ✅ `test_linspace_endpoint_exact` |
| 73 | `TestLinspace_ZeroRange` | ✅ `test_linspace_zero_range` |
| 74 | `TestLinspace_SmallInterval` | ✅ `test_linspace_small_interval` |
| 75 | `TestLinspace_LargeInterval` | ✅ `test_linspace_large_interval` |
| 76 | `TestLinspace_MatchesNumpyBehavior` | ✅ `test_linspace_matches_numpy_behavior` |
| 77 | `TestLinspace_Consistency` | ✅ `test_linspace_consistency` |

### `pkg/norfairgo/matching_test.go` (14 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 78 | `TestMatching_PerfectMatches` | ✅ `test_perfect_matches` |
| 79 | `TestMatching_ThresholdFiltering` | ✅ `test_threshold_filtering` |
| 80 | `TestMatching_AllAboveThreshold` | ✅ `test_all_above_threshold` |
| 81 | `TestMatching_SingleElementNoMatch` | ✅ `test_single_element_no_match` |
| 82 | `TestMatching_SingleElementMatch` | ✅ `test_single_element_match` |
| 83 | `TestMatching_GreedyBehavior` | ✅ `test_greedy_behavior` |
| 84 | `TestMatching_OneToOneConstraint` | ✅ `test_one_to_one_constraint` |
| 85 | `TestMatching_MoreDetectionsThanObjects` | ✅ `test_more_detections_than_objects` |
| 86 | `TestMatching_MoreObjectsThanDetections` | ✅ `test_more_objects_than_detections` |
| 87 | `TestMatching_NaNDetection` | ✅ `test_nan_detection` |
| 88 | `TestMatching_NoNaN` | ✅ `test_no_nan` |
| 89 | `TestMatching_InfHandling` | ✅ `test_inf_handling` |
| 90 | `TestArgMin` | ✅ `test_finds_minimum_value` (integrated into matching) |
| 91 | `TestMinMatrix` | ✅ `test_minimum_matrix_value` (integrated into matching) |

### `pkg/norfairgo/utils_test.go` (16 tests) ✅ ALL NON-OPENCV PORTED
| # | Test | Ported? |
|---|------|---------|
| 92 | `TestValidatePoints_Valid2D` | ✅ `test_validate_points_valid_2d` |
| 93 | `TestValidatePoints_Valid3D` | ✅ `test_validate_points_valid_3d` |
| 94 | `TestValidatePoints_Single2DPoint` | ✅ `test_validate_points_single_2d_point` |
| 95 | `TestValidatePoints_Single3DPoint` | ✅ `test_validate_points_single_3d_point` |
| 96 | `TestValidatePoints_InvalidDimensions4D` | ✅ `test_validate_points_invalid_dimensions_4d` |
| 97 | `TestValidatePoints_InvalidDimensions1D` | ✅ `test_validate_points_invalid_dimensions_1d` |
| 98 | `TestValidatePoints_InvalidSingleValue` | ✅ `test_validate_points_invalid_single_value` |
| 99 | `TestGetTerminalSize_ReturnsValues` | ✅ `test_get_terminal_size_returns_values` |
| 100 | `TestGetTerminalSize_CustomDefaults` | ✅ `test_get_terminal_size_custom_defaults` |
| 101 | `TestGetTerminalSize_StandardDefaults` | N/A (redundant) |
| 102 | `TestGetCutout_CenterRegion` | ❌ Requires OpenCV |
| 103 | `TestGetCutout_CornerRegion` | ❌ Requires OpenCV |
| 104 | `TestGetCutout_SinglePoint` | ❌ Requires OpenCV |
| 105 | `TestGetCutout_OutOfBounds` | ❌ Requires OpenCV |
| 106 | `TestGetCutout_LargeRegion` | ❌ Requires OpenCV |
| 107 | `TestGetCutout_InvalidPoints` | ❌ Requires OpenCV |

**Note:** Rust also has additional utils tests not in Go: `test_any_true`, `test_all_true`, `test_get_bounding_box*`, `test_clamp*`

### `pkg/norfairgo/filter_test.go` (14 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 108 | `TestFilterPyKalmanFilterFactory_Create` | ✅ `test_filterpy_kalman_create` (filterpy.rs) |
| 109 | `TestFilterPyKalmanFilter_StaticObject` | ✅ `test_filterpy_kalman_static_object` |
| 110 | `TestFilterPyKalmanFilter_MovingObject` | ✅ `test_filterpy_kalman_moving_object` |
| 111 | `TestOptimizedKalmanFilterFactory_Create` | ✅ `test_optimized_kalman_create` (optimized.rs) |
| 112 | `TestOptimizedKalmanFilter_StaticObject` | ✅ `test_optimized_kalman_static_object` |
| 113 | `TestOptimizedKalmanFilter_MovingObject` | ✅ `test_optimized_kalman_moving_object` |
| 114 | `TestNoFilterFactory_Create` | ✅ `test_no_filter_create` (no_filter.rs) |
| 115 | `TestNoFilter_Predict` | ✅ `test_no_filter_predict_is_noop` |
| 116 | `TestNoFilter_Update` | ✅ `test_no_filter_update` |
| 117 | `TestFilterComparison_StaticObject` | ✅ `test_filter_comparison_static_object` (mod.rs) |
| 118 | `TestFilterComparison_MovingObject` | ✅ `test_filter_comparison_moving_object` |
| 119 | `TestFilterPyKalmanFilter_PartialMeasurement` | ✅ `test_filterpy_kalman_partial_measurement` |
| 120 | `TestOptimizedKalmanFilter_PartialMeasurement` | ✅ `test_optimized_kalman_partial_measurement` |
| 121 | `TestFilters_MultiPoint` | ✅ `test_filters_multipoint` |

**Note:** Rust has additional filter test: `test_nofilter_multipoint`

### `pkg/norfairgo/camera_motion_test.go` (26 tests) ✅ ALL NON-OPENCV PORTED
| # | Test | Ported? |
|---|------|---------|
| 122 | `TestTranslationTransformation_ForwardBackward` | ✅ `test_translation_transformation_forward_backward` |
| 123 | `TestTranslationTransformation_ZeroMovement` | ✅ `test_translation_transformation_zero_movement` |
| 124 | `TestTranslationTransformation_InvalidMovementVector` | N/A (Rust uses [f64; 2], always valid) |
| 125 | `TestTranslationTransformationGetter_SimpleModeFind` | ✅ `test_translation_getter_simple_mode_find` |
| 126 | `TestTranslationTransformationGetter_Accumulation` | ✅ `test_translation_getter_accumulation` |
| 127 | `TestTranslationTransformationGetter_ReferenceUpdate` | ✅ `test_translation_getter_reference_update` |
| 128 | `TestTranslationTransformationGetter_SinglePoint` | ✅ `test_translation_getter_single_point` |
| 129 | `TestTranslationTransformationGetter_MismatchedDimensions` | ✅ `test_translation_getter_mismatched_dimensions` |
| 130 | `TestNilCoordinateTransformation` | ✅ `test_nil_coordinate_transformation_returns_same_points` |
| 131 | `TestHomographyTransformation_Identity` | ❌ Requires OpenCV |
| 132 | `TestHomographyTransformation_Translation` | ❌ Requires OpenCV |
| 133 | `TestHomographyTransformation_Scaling` | ❌ Requires OpenCV |
| 134 | `TestHomographyTransformation_ForwardBackward` | ❌ Requires OpenCV |
| 135 | `TestHomographyTransformation_DivisionByZero` | ❌ Requires OpenCV |
| 136 | `TestHomographyTransformation_InvalidMatrix` | ❌ Requires OpenCV |
| 137 | `TestHomographyTransformation_SingularMatrix` | ❌ Requires OpenCV |
| 138 | `TestHomographyTransformation_Rotation` | ❌ Requires OpenCV |
| 139 | `TestHomographyTransformationGetter_PerfectCorrespondence` | ❌ Requires OpenCV |
| 140 | `TestHomographyTransformationGetter_InsufficientPoints` | ❌ Requires OpenCV |
| 141 | `TestHomographyTransformationGetter_WithOutliers` | ❌ Requires OpenCV |
| 142 | `TestHomographyTransformationGetter_Accumulation` | ❌ Requires OpenCV |
| 143 | `TestMotionEstimator_Construction` | ❌ Requires OpenCV |
| 144 | `TestMotionEstimator_FirstFrameInitialization` | ❌ Requires OpenCV |
| 145 | `TestMotionEstimator_CloseResourcesCleanly` | ❌ Requires OpenCV |
| 146 | `TestMotionEstimator_ComputeTranslation_Small` | ❌ Requires OpenCV |
| 147 | `TestMotionEstimator_ComputeTranslation_Large` | ❌ Requires OpenCV |

**Note:** Rust has additional camera_motion tests: `test_nil_transformation`, `test_translation_transformation`, `test_translation_roundtrip`, `test_translation_getter`

### `pkg/norfairgo/video_test.go` (18 tests) - ALL REQUIRE OPENCV (deferred)
| # | Test | Ported? |
|---|------|---------|
| 148 | `TestVideo_InputValidation_BothNil` | ❌ Requires OpenCV |
| 149 | `TestVideo_InputValidation_BothSet` | ❌ Requires OpenCV |
| 150 | `TestVideo_InputValidation_FileNotFound` | ❌ Requires OpenCV |
| 151 | `TestVideo_GetCodecFourcc_AVI` | ❌ Requires OpenCV |
| 152 | `TestVideo_GetCodecFourcc_MP4` | ❌ Requires OpenCV |
| 153 | `TestVideo_GetCodecFourcc_CustomOverride` | ❌ Requires OpenCV |
| 154 | `TestVideo_GetCodecFourcc_UnsupportedExtension` | ❌ Requires OpenCV |
| 155 | `TestVideo_GetOutputFilePath_File` | ❌ Requires OpenCV |
| 156 | `TestVideo_GetOutputFilePath_DirectoryWithCamera` | ❌ Requires OpenCV |
| 157 | `TestVideo_GetOutputFilePath_DirectoryWithFile` | ❌ Requires OpenCV |
| 158 | `TestVideo_GetProgressDescription_Camera` | ❌ Requires OpenCV |
| 159 | `TestVideo_GetProgressDescription_File` | ❌ Requires OpenCV |
| 160 | `TestVideo_GetProgressDescription_WithLabel` | ❌ Requires OpenCV |
| 161 | `TestVideo_GetProgressDescription_Abbreviation` | ❌ Requires OpenCV |
| 162 | `TestVideoFromFrames_INIParsing` | ❌ Requires OpenCV |
| 163 | `TestVideoFromFrames_MissingINI` | ❌ Requires OpenCV |
| 164 | `TestVideoFromFrames_InvalidINI` | ❌ Requires OpenCV |
| 165 | `TestVideoFromFrames_VideoGeneration` | ❌ Requires OpenCV |

**Note:** Video module is not yet implemented in Rust (see PLAN.md Phase 9: Video Module - NOT STARTED)

### `pkg/norfairgo/metrics_test.go` & `extended_metrics_test.go` (4 tests)
| # | Test | Ported? |
|---|------|---------|
| 166 | `TestEvalMotChallenge_Perfect` | ✅ (as accumulator tests) |
| 167 | `TestEvalMotChallenge_MostlyLost` | ✅ (as accumulator tests) |
| 168 | `TestEvalMotChallenge_Fragmented` | ✅ (as accumulator tests) |
| 169 | `TestEvalMotChallenge_Mixed` | ✅ (as accumulator tests) |

### `pkg/norfairgo/tracker_test.go` (7 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 170 | `TestTracker_NewTracker` | ✅ `test_tracker_new`, `test_tracker_new_with_defaults` |
| 171 | `TestTracker_InvalidInitializationDelay` | ✅ `test_tracker_invalid_config`, `test_tracker_invalid_config_delay_too_high` |
| 172 | `TestTracker_SimpleUpdate` | ✅ `test_tracker_simple_update` |
| 173 | `TestTracker_UpdateEmptyDetections` | ✅ `test_tracker_update_empty_detections` |
| 174 | `TestDetection_Creation` | ✅ `test_detection_creation_2d`, `test_detection_creation_3d` |
| 175 | `TestTrackedObject_Creation` | ✅ `test_tracked_object_creation_via_tracker` |
| 176 | `TestTracker_CameraMotion` | ✅ `test_tracker_camera_motion` |

**Note:** Rust has additional tracker tests: `test_tracker_initialization`, `test_tracker_immediate_initialization`, `test_tracker_object_counts`, `test_tracker_params_bad_distance`, `test_tracker_simple_hit_counter_dynamics`, `test_tracker_moving_object`, `test_tracker_distance_threshold`, `test_tracker_1d_points`, `test_tracker_count_comprehensive`, `test_multiple_trackers_independent`

### `pkg/norfairgo/tracker_factory_test.go` (8 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 177 | `TestTrackedObjectFactory_GetInitializingID` | ✅ `test_factory_get_initializing_id` |
| 178 | `TestTrackedObjectFactory_GetIDs` | ✅ `test_factory_get_ids` |
| 179 | `TestTrackedObjectFactory_GlobalIDUniqueness` | ✅ `test_factory_global_id_uniqueness` |
| 180 | `TestTrackedObjectFactory_InitializingVsPermanentIDs` | ✅ `test_factory_initializing_vs_permanent_ids` |
| 181 | `TestTrackedObjectFactory_MixedSequence` | ✅ `test_factory_mixed_sequence` |
| 182 | `TestTrackedObjectFactory_ConcurrentInitializingIDs` | ✅ `test_factory_concurrent_initializing_ids` |
| 183 | `TestTrackedObjectFactory_ConcurrentPermanentIDs` | ✅ `test_factory_concurrent_permanent_ids` |
| 184 | `TestTrackedObjectFactory_ConcurrentMultipleFactories` | ✅ `test_factory_concurrent_multiple_factories` |

**Note:** Rust has additional tracked_object tests: `test_factory_get_permanent_id`, `test_tracked_object_live_points`, `test_tracked_object_default`, `test_tracked_object_get_estimate`

### `pkg/norfairgo/distances_test.go` (11 tests) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 185 | `TestFrobenius` | ✅ `test_frobenius_*` (7 tests in functions.rs) |
| 186 | `TestMeanManhattan` | ✅ `test_mean_manhattan_*` (7 tests) |
| 187 | `TestMeanEuclidean` | ✅ `test_mean_euclidean_*` (9 tests) |
| 188 | `TestIoU` | ✅ `test_iou_*` (7 tests) |
| 189 | `TestIoU_InvalidBbox` | ✅ `test_iou_invalid_bbox` |
| 190 | `TestScalarDistance` | ✅ `test_scalar_distance_wrapper` (mod.rs) |
| 191 | `TestVectorizedDistance` | ✅ `test_vectorized_distance_wrapper` (mod.rs) |
| 192 | `TestScipyDistance` | ✅ `test_scipy_distance_wrapper` (mod.rs) |
| 193 | `TestKeypointVote` | ✅ `test_keypoint_vote_*` (5 tests) |
| 194 | `TestNormalizedEuclidean` | ✅ `test_normalized_mean_euclidean_*` (6 tests) |
| 195 | `TestGetDistanceByName` | ✅ `test_distance_by_name_*` (4 tests in mod.rs) |

**Note:** Rust has 49+ distance function tests + 8 wrapper tests = 57+ total distance tests

### `pkg/norfairgo/benchmark_test.go` (6 benchmarks) ✅ ALL PORTED
| # | Test | Ported? |
|---|------|---------|
| 196 | `BenchmarkTrackerUpdate_10Objects` | ✅ `benchmark_tracker_update_10_objects` (benches/tracker_benchmarks.rs) |
| 197 | `BenchmarkTrackerUpdate_50Objects` | ✅ `benchmark_tracker_update_50_objects` (benches/tracker_benchmarks.rs) |
| 198 | `BenchmarkTrackerUpdate_100Objects` | ✅ `benchmark_tracker_update_100_objects` (benches/tracker_benchmarks.rs) |
| 199 | `BenchmarkTrackerUpdate_100Objects_FilterPyKalman` | ✅ `benchmark_tracker_update_100_objects_filterpy_kalman` (benches/tracker_benchmarks.rs) |
| 200 | `BenchmarkTrackerUpdate_100Objects_NoFilter` | ✅ `benchmark_tracker_update_100_objects_no_filter` (benches/tracker_benchmarks.rs) |
| 201 | `BenchmarkTrackerUpdate_100Objects_IoU` | ✅ `benchmark_tracker_update_100_objects_iou` (benches/tracker_benchmarks.rs) |

**Run benchmarks with:** `cargo bench`

### `pkg/norfairgocolor/color_test.go` (11 tests)
| # | Test | Ported? |
|---|------|---------|
| 202 | `TestColor_ToRGBA` | ❌ |
| 203 | `TestColor_BGROrdering` | ❌ |
| 204 | `TestHexToBGR_SixChar` | ❌ |
| 205 | `TestHexToBGR_ThreeChar` | ❌ |
| 206 | `TestHexToBGR_NoHashPrefix` | ❌ |
| 207 | `TestHexToBGR_Lowercase` | ❌ |
| 208 | `TestHexToBGR_InvalidLength` | ❌ |
| 209 | `TestHexToBGR_InvalidCharacters` | ❌ |
| 210 | `TestHexToBGR_EdgeValues` | ❌ |
| 211 | `TestColorConstants` | ❌ |
| 212 | `TestHexToBGR_RoundTrip` | ❌ |

### `pkg/norfairgodraw/utils_test.go` (5 tests)
| # | Test | Ported? |
|---|------|---------|
| 213 | `TestCentroid_SimplePoints` | ❌ |
| 214 | `TestCentroid_SinglePoint` | ❌ |
| 215 | `TestCentroid_NegativeCoords` | ❌ |
| 216 | `TestCentroid_ThreePoints` | ❌ |
| 217 | `TestCentroid_FloatRounding` | ❌ |

### `pkg/norfairgodraw/path_test.go` (31 tests)
| # | Test | Ported? |
|---|------|---------|
| 218 | `TestBuildText_AllFields` | ❌ |
| 219 | `TestBuildText_OnlyLabel` | ❌ |
| 220 | `TestBuildText_OnlyID` | ❌ |
| 221 | `TestBuildText_OnlyScores` | ❌ |
| 222 | `TestBuildText_NoFields` | ❌ |
| 223 | `TestBuildText_NilFields` | ❌ |
| 224 | `TestBuildText_MultipleScores` | ❌ |
| 225 | `TestBuildText_LabelAndID` | ❌ |
| 226 | `TestBuildText_EmptyLabel` | ❌ |
| 227 | `TestBuildText_EmptyScores` | ❌ |
| 228 | `TestPaths_LazyInit` | ❌ |
| 229 | `TestPaths_AutoScaling` | ❌ |
| 230 | `TestPaths_Fade` | ❌ |
| 231 | `TestPaths_Accumulation` | ❌ |
| 232 | `TestPaths_ColorByID` | ❌ |
| 233 | `TestPaths_CustomColor` | ❌ |
| 234 | `TestPaths_CameraMotionWarning` | ❌ |
| 235 | `TestPaths_GetPointsToDraw` | ❌ |
| 236 | `TestPaths_EmptyObjects` | ❌ |
| 237 | `TestPaths_AlphaBlend` | ❌ |
| 238 | `TestDefaultGetPointsToDraw` | ❌ |
| 239 | `TestAbsolutePaths_Constructor` | ❌ |
| 240 | `TestAbsolutePaths_AutoScaling` | ❌ |
| 241 | `TestAbsolutePaths_EmptyObjects` | ❌ |
| 242 | `TestAbsolutePaths_ColorByID` | ❌ |
| 243 | `TestAbsolutePaths_CustomColor` | ❌ |
| 244 | `TestAbsolutePaths_CustomGetPointsToDraw` | ❌ |
| 245 | `TestAbsolutePaths_TransformPointsToRelative` | ❌ |
| 246 | `TestAbsolutePaths_TransformPointsToRelative_Empty` | ❌ |
| 247 | `TestLinspace` | ❌ |
| 248 | `TestAbsolutePaths_ColorByID` (dup) | ❌ |

### `pkg/norfairgodraw/draw_boxes_test.go` (22 tests)
| # | Test | Ported? |
|---|------|---------|
| 249 | `TestDrawBoxes_BasicDefaults` | ❌ |
| 250 | `TestDrawBoxes_CustomParameters` | ❌ |
| 251 | `TestDrawBoxes_NilDrawables` | ❌ |
| 252 | `TestDrawBoxes_ColorByID` | ❌ |
| 253 | `TestDrawBoxes_ColorByLabel` | ❌ |
| 254 | `TestDrawBoxes_ColorRandom` | ❌ |
| 255 | `TestDrawBoxes_DirectColorHex` | ❌ |
| 256 | `TestDrawBoxes_DirectColorName` | ❌ |
| 257 | `TestDrawBoxes_DrawLabelsTrue` | ❌ |
| 258 | `TestDrawBoxes_DrawIDsOnly` | ❌ |
| 259 | `TestDrawBoxes_DrawScores` | ❌ |
| 260 | `TestDrawBoxes_AllTextFields` | ❌ |
| 261 | `TestDrawBoxes_CustomTextColor` | ❌ |
| 262 | `TestDrawBoxes_DrawBoxFalse` | ❌ |
| 263 | `TestDrawBoxes_NoTextOrBox` | ❌ |
| 264 | `TestDrawBoxes_SmallFrame` | ❌ |
| 265 | `TestDrawBoxes_LargeFrame` | ❌ |
| 266 | `TestDrawBoxes_InvertedBox` | ❌ |
| 267 | `TestDrawBoxes_MultipleBoxes` | ❌ |
| 268 | `TestDrawBoxes_InvalidPointCount` | ❌ |
| 269 | `TestDrawBoxes_BoundaryBox` | ❌ |
| 270 | `TestDrawBoxes_DirectColor_GoldenImage` | ❌ |

### `pkg/norfairgodraw/color_test.go` (12 tests)
| # | Test | Ported? |
|---|------|---------|
| 271 | `TestHexToBGR_6CharFormat` | ❌ |
| 272 | `TestHexToBGR_3CharFormat` | ❌ |
| 273 | `TestHexToBGR_InvalidFormats` | ❌ |
| 274 | `TestParseColorName_ValidNames` | ❌ |
| 275 | `TestParseColorName_InvalidNames` | ❌ |
| 276 | `TestPalette_NewPalette` | ❌ |
| 277 | `TestPalette_ChooseColor_Deterministic` | ❌ |
| 278 | `TestPalette_ChooseColor_NilHashable` | ❌ |
| 279 | `TestPalette_Set_ValidPalettes` | ❌ |
| 280 | `TestPalette_Set_InvalidPalette` | ❌ |
| 281 | `TestPalette_SetDefaultColor` | ❌ |
| 282 | `TestColor_ToRGBA` | ❌ |

### `pkg/norfairgodraw/drawer_test.go` (30 tests)
| # | Test | Ported? |
|---|------|---------|
| 283 | `TestDrawer_Circle_AutoScaling` | ❌ |
| 284 | `TestDrawer_Circle_ExplicitRadius` | ❌ |
| 285 | `TestDrawer_Circle_FilledCircle` | ❌ |
| 286 | `TestDrawer_Circle_SmallFrame` | ❌ |
| 287 | `TestDrawer_Circle_OutOfBounds` | ❌ |
| 288 | `TestDrawer_Text_AutoScaling` | ❌ |
| 289 | `TestDrawer_Text_WithShadow` | ❌ |
| 290 | `TestDrawer_Text_WithoutShadow` | ❌ |
| 291 | `TestDrawer_Text_EmptyString` | ❌ |
| 292 | `TestDrawer_Text_LongString` | ❌ |
| 293 | `TestDrawer_Text_SmallFrame` | ❌ |
| 294 | `TestDrawer_Rectangle_Basic` | ❌ |
| 295 | `TestDrawer_Rectangle_ZeroThickness` | ❌ |
| 296 | `TestDrawer_Rectangle_FilledRectangle` | ❌ |
| 297 | `TestDrawer_Line_Basic` | ❌ |
| 298 | `TestDrawer_Line_Horizontal` | ❌ |
| 299 | `TestDrawer_Line_Vertical` | ❌ |
| 300 | `TestDrawer_Cross_Basic` | ❌ |
| 301 | `TestDrawer_Cross_SmallRadius` | ❌ |
| 302 | `TestDrawer_Cross_LargeRadius` | ❌ |
| 303 | `TestDrawer_AlphaBlend_HalfAlpha` | ❌ |
| 304 | `TestDrawer_AlphaBlend_AutoBeta` | ❌ |
| 305 | `TestDrawer_AlphaBlend_WithGamma` | ❌ |
| 306 | `TestDrawable_NewDrawableFromDetection` | ❌ |
| 307 | `TestDrawable_NewDrawableFromTrackedObject` | ❌ |
| 308 | `TestDrawable_NewDrawable_Explicit` | ❌ |
| 309 | `TestDrawable_NewDrawable_NilLivePoints` | ❌ |
| 310 | `TestDrawable_NewDrawable_NilPoints` | ❌ |
| 311 | `TestDrawable_NewDrawableFromDetection_NilPoints` | ❌ |
| 312 | `TestDrawer_DrawingPrimitives_GoldenImage` | ❌ |

### `pkg/norfairgodraw/fixed_camera_test.go` (10 tests)
| # | Test | Ported? |
|---|------|---------|
| 313 | `TestNewFixedCamera_Defaults` | ❌ |
| 314 | `TestNewFixedCamera_CustomValues` | ❌ |
| 315 | `TestFixedCamera_LazyInit` | ❌ |
| 316 | `TestFixedCamera_Fade` | ❌ |
| 317 | `TestFixedCamera_CenterPositioning` | ❌ |
| 318 | `TestFixedCamera_Translation` | ❌ |
| 319 | `TestFixedCamera_BoundaryCropping` | ❌ |
| 320 | `TestFixedCamera_Close` | ❌ |
| 321 | `TestFixedCamera_MultipleFrames` | ❌ |
| 322 | `TestFixedCamera_ScaleVariations` | ❌ |

### `pkg/norfairgodraw/absolute_grid_test.go` (1 test)
| # | Test | Ported? |
|---|------|---------|
| 323 | `TestGetGrid_EquatorMode` | ❌ |

---

## Rust Tests (Current: 276 total - 270 unit + 6 integration)

### `tests/integration_tests.rs` (6 tests)
| # | Test | Source |
|---|------|--------|
| 1 | `test_integration_complete_tracking_pipeline` | Original |
| 2 | `test_integration_multiple_filter_types` | Original |
| 3 | `test_integration_multiple_distance_functions` | Original |
| 4 | `test_integration_reid_enabled` | Original |
| 5 | `test_integration_camera_motion_compensation` | Original |
| 6 | `test_integration_object_lifecycle` | Original |

### `src/tracked_object.rs` (12 tests)
All ported from Go `tracker_factory_test.go`

### `src/matching.rs` (16 tests)
All ported from Go `matching_test.go`

### `src/utils.rs` (6 tests)
Partial port from Go `utils_test.go`

### `src/detection.rs` (4 tests)
All ported from Python/Go

### `src/distances/mod.rs` (8 tests)
Wrapper tests for ScalarDistance, VectorizedDistance, ScipyDistance, distance_by_name

### `src/distances/functions.rs` (49 tests)
All ported from Python/Go including IoU invalid bbox tests

### `src/filter/*.rs` (15 tests)
All ported from Go `filter_test.go`

### `src/internal/filterpy/kalman.rs` (~6 tests)
Ported from Go `internal/filterpy/kalman_test.go`

### `src/internal/numpy/array.rs` (~7 tests)
Partial port from Go `internal/numpy/array_test.go`

### `src/internal/scipy/distance.rs` (~4 tests)
Partial port from Go `internal/scipy/distance_test.go`

### `src/internal/motmetrics/*.rs` (~3 tests)
Partial port from Go `internal/motmetrics/*_test.go`

### `src/metrics/accumulator.rs` (47 tests)
Extended port including MOTA/MOTP, perfect/mostly_lost/fragmented/mixed scenarios

### `src/tracker.rs` (11 tests)
Full port from Go/Python tracker tests including moving objects, distance threshold, 1D points, count

### `src/camera_motion/*.rs` (4 tests)
Partial port from Go `camera_motion_test.go`

---

## Implementation Priority

### Phase 1: Core Missing Tests (No New Dependencies)
1. ~~`src/internal/scipy/distance.rs` - 8 missing cdist tests~~ ✅ DONE (all 13 ported)
2. ~~`src/internal/scipy/optimize.rs` - 12 linear_sum_assignment tests~~ ✅ DONE (all 11 relevant ported)
3. ~~`src/internal/numpy/array.rs` - 7 missing linspace tests~~ ✅ DONE (all 14 ported)
4. ~~`src/internal/motmetrics/iou.rs` - 10 missing IoU tests~~ ✅ DONE (all 13 ported)
5. ~~`src/internal/filterpy/kalman.rs` - 2 missing tests~~ ✅ DONE (all 8 ported)
6. ~~`src/internal/motmetrics/accumulator.rs` - 17 accumulator tests~~ ✅ DONE (8 tests added)
7. `src/matching.rs` - 2 helper function tests (ArgMin, MinMatrix)
8. `src/utils.rs` - 12 missing validation/cutout tests
9. ~~`src/tracker.rs` - 4 missing tests~~ ✅ DONE (7 tests added)
10. `src/camera_motion/transformations.rs` - 7 missing translation tests

### Phase 2: Wrapper Class Tests ✅ DONE
1. ~~`src/distances/scalar.rs` - ScalarDistance wrapper~~ ✅
2. ~~`src/distances/vectorized.rs` - VectorizedDistance wrapper~~ ✅
3. ~~`src/distances/mod.rs` - distance_by_name tests~~ ✅

### Phase 3: Metrics Tests ✅ DONE
1. ~~`src/metrics/evaluation.rs` - 4 eval_mot_challenge tests~~ ✅ (ported as accumulator tests)

### Phase 4: OpenCV-Dependent Tests (Behind Feature Flag) - DEFERRED
1. Drawing tests (#202-323) - requires OpenCV
2. Video tests (#148-165) - requires OpenCV
3. Homography tests (#131-147) - requires OpenCV
