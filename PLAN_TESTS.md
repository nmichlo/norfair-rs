# Test Porting Checklist

Complete list of ALL tests from Python, Go, and their port status to Rust.

## RULES

re-read .claude/CLAUDE.md, then continue porting tests from PLAN_TESTS.md (except drawing tests). DO NOT MISS ANY TESTS. If you think you can skip tests, you are wrong, implement them. If you want to skip
anything, you are wrong, do not skip it. If you notice logic differences according between rust/python/go code, you MUST fix those errors (python is the source of truth, golang has extra tests and is easier to
port to rust but not 100% trustworthy, rust should not be trusted this is what we are building and fixing). You must ALWAYS follow the process in the GOLDEN RULE of claude.md when carrying out these tasks.

## Summary Statistics

| Language | Total Tests | Ported | Missing |
|----------|-------------|--------|---------|
| Python | 19 | 12 | 7 |
| Go | 323 | ~90 | ~233 |
| **Total** | 342 | ~102 | ~240 |

**Note:** Go tests #99-101 (Terminal) and drawing tests (#213-323) require OpenCV and can be deferred.

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
| 8 | `test_scalar_distance` | ❌ |
| 9 | `test_vectorized_distance` | ❌ |
| 10 | `test_scipy_distance` | ✅ |

### `tests/test_tracker.py` (9 tests)
| # | Test | Ported? |
|---|------|---------|
| 11 | `test_params` | ✅ |
| 12 | `test_simple` | ✅ |
| 13 | `test_moving` | ⚠️ partial |
| 14 | `test_distance_t` | ❌ |
| 15 | `test_1d_points` | ❌ |
| 16 | `test_camera_motion` | ✅ |
| 17 | `test_count` | ❌ |
| 18 | `test_multiple_trackers` | ✅ |
| 19 | `test_reid_hit_counter` | ⚠️ partial |

---

## Go Tests (~323 total)

### `internal/scipy/distance_test.go` (13 tests)
| # | Test | Ported? |
|---|------|---------|
| 1 | `TestCdist_Euclidean` | ✅ |
| 2 | `TestCdist_Manhattan` | ✅ |
| 3 | `TestCdist_Cosine` | ✅ |
| 4 | `TestCdist_Cosine_Parallel` | ❌ |
| 5 | `TestCdist_Cosine_ZeroVector` | ❌ |
| 6 | `TestCdist_SquaredEuclidean` | ❌ |
| 7 | `TestCdist_Chebyshev` | ✅ |
| 8 | `TestCdist_DifferentDimensions` | ❌ |
| 9 | `TestCdist_PanicOnMismatchedColumns` | ❌ |
| 10 | `TestCdist_PanicOnUnsupportedMetric` | ❌ |
| 11 | `TestCdist_SingleVectors` | ❌ |
| 12 | `TestCdist_IdenticalVectors` | ❌ |
| 13 | `TestCdist_Cosine_AntiParallel` | ❌ |

### `internal/scipy/optimize_test.go` (12 tests)
| # | Test | Ported? |
|---|------|---------|
| 14 | `TestLinearSumAssignment_BasicSquare` | ❌ |
| 15 | `TestLinearSumAssignment_CostThreshold` | ❌ |
| 16 | `TestLinearSumAssignment_RectangularMoreRows` | ❌ |
| 17 | `TestLinearSumAssignment_RectangularMoreCols` | ❌ |
| 18 | `TestLinearSumAssignment_EmptyMatrix` | ❌ |
| 19 | `TestLinearSumAssignment_EmptyColumns` | ❌ |
| 20 | `TestLinearSumAssignment_AllRejectedByThreshold` | ❌ |
| 21 | `TestLinearSumAssignment_OptimalMatching` | ❌ |
| 22 | `TestLinearSumAssignment_SingleElement` | ❌ |
| 23 | `TestLinearSumAssignment_PartialMatching` | ❌ |
| 24 | `TestLinearSumAssignment_ZeroCosts` | ❌ |
| 25 | `TestLinearSumAssignment_max_helper` | ❌ |

### `internal/filterpy/kalman_test.go` (8 tests)
| # | Test | Ported? |
|---|------|---------|
| 26 | `TestNewKalmanFilter` | ✅ |
| 27 | `TestKalmanFilter_Predict` | ✅ |
| 28 | `TestKalmanFilter_Update` | ✅ |
| 29 | `TestKalmanFilter_PredictUpdateCycle` | ✅ |
| 30 | `TestKalmanFilter_PartialMeasurement` | ✅ |
| 31 | `TestKalmanFilter_SingularInnovationCovariance` | ❌ |
| 32 | `TestKalmanFilter_GettersSetters` | ⚠️ partial |
| 33 | `TestKalmanFilter_MultiDimensional` | ✅ |

### `internal/motmetrics/iou_test.go` (13 tests)
| # | Test | Ported? |
|---|------|---------|
| 34 | `TestIouDistance_PerfectOverlap` | ✅ |
| 35 | `TestIouDistance_NoOverlap` | ✅ |
| 36 | `TestIouDistance_PartialOverlap` | ✅ |
| 37 | `TestIouDistance_ContainedBox` | ❌ |
| 38 | `TestIouDistance_AdjacentBoxes` | ❌ |
| 39 | `TestIouDistance_SmallOverlap` | ❌ |
| 40 | `TestIouDistance_FloatingPoint` | ❌ |
| 41 | `TestIouDistance_InvalidBox1` | ❌ |
| 42 | `TestIouDistance_InvalidBox2` | ❌ |
| 43 | `TestIouDistance_WrongLength` | ❌ |
| 44 | `TestComputeIoUMatrix` | ❌ |
| 45 | `TestComputeIoUMatrix_Empty` | ❌ |
| 46 | `TestComputeIoUMatrix_LargeSet` | ❌ |

### `internal/motmetrics/accumulator_test.go` (17 tests)
| # | Test | Ported? |
|---|------|---------|
| 47 | `TestNewTrackLifecycle` | ❌ |
| 48 | `TestTrackLifecycle_UpdateMatched` | ❌ |
| 49 | `TestTrackLifecycle_UpdateMissed` | ❌ |
| 50 | `TestTrackLifecycle_Fragmentation` | ❌ |
| 51 | `TestTrackLifecycle_Coverage` | ❌ |
| 52 | `TestNewMOTAccumulator` | ❌ |
| 53 | `TestMOTAccumulator_Update_EmptyFrame` | ❌ |
| 54 | `TestMOTAccumulator_Update_OnlyPredictions` | ❌ |
| 55 | `TestMOTAccumulator_Update_OnlyGT` | ❌ |
| 56 | `TestMOTAccumulator_Update_PerfectMatch` | ❌ |
| 57 | `TestMOTAccumulator_Update_PartialMatch` | ❌ |
| 58 | `TestMOTAccumulator_DetectSwitches` | ❌ |
| 59 | `TestMOTAccumulator_MultiFrame` | ❌ |
| 60 | `TestComputeExtendedMetrics_MostlyTracked` | ❌ |
| 61 | `TestComputeExtendedMetrics_MostlyLost` | ❌ |
| 62 | `TestComputeExtendedMetrics_PartiallyTracked` | ❌ |
| 63 | `TestComputeExtendedMetrics_Fragmentations` | ❌ |

### `internal/numpy/array_test.go` (14 tests)
| # | Test | Ported? |
|---|------|---------|
| 64 | `TestLinspace_Basic` | ✅ |
| 65 | `TestLinspace_TwoPoints` | ✅ |
| 66 | `TestLinspace_SinglePoint` | ✅ |
| 67 | `TestLinspace_Zero` | ✅ |
| 68 | `TestLinspace_Negative` | ✅ |
| 69 | `TestLinspace_ReverseRange` | ✅ |
| 70 | `TestLinspace_FloatingPoint` | ✅ |
| 71 | `TestLinspace_LargeN` | ❌ |
| 72 | `TestLinspace_EndpointExact` | ❌ |
| 73 | `TestLinspace_ZeroRange` | ❌ |
| 74 | `TestLinspace_SmallInterval` | ❌ |
| 75 | `TestLinspace_LargeInterval` | ❌ |
| 76 | `TestLinspace_MatchesNumpyBehavior` | ❌ |
| 77 | `TestLinspace_Consistency` | ❌ |

### `pkg/norfairgo/matching_test.go` (14 tests)
| # | Test | Ported? |
|---|------|---------|
| 78 | `TestMatching_PerfectMatches` | ✅ |
| 79 | `TestMatching_ThresholdFiltering` | ✅ |
| 80 | `TestMatching_AllAboveThreshold` | ✅ |
| 81 | `TestMatching_SingleElementNoMatch` | ✅ |
| 82 | `TestMatching_SingleElementMatch` | ✅ |
| 83 | `TestMatching_GreedyBehavior` | ✅ |
| 84 | `TestMatching_OneToOneConstraint` | ✅ |
| 85 | `TestMatching_MoreDetectionsThanObjects` | ✅ |
| 86 | `TestMatching_MoreObjectsThanDetections` | ✅ |
| 87 | `TestMatching_NaNDetection` | ✅ |
| 88 | `TestMatching_NoNaN` | ✅ |
| 89 | `TestMatching_InfHandling` | ✅ |
| 90 | `TestArgMin` | ❌ |
| 91 | `TestMinMatrix` | ❌ |

### `pkg/norfairgo/utils_test.go` (16 tests)
| # | Test | Ported? |
|---|------|---------|
| 92 | `TestValidatePoints_Valid2D` | ✅ |
| 93 | `TestValidatePoints_Valid3D` | ❌ |
| 94 | `TestValidatePoints_Single2DPoint` | ❌ |
| 95 | `TestValidatePoints_Single3DPoint` | ❌ |
| 96 | `TestValidatePoints_InvalidDimensions4D` | ❌ |
| 97 | `TestValidatePoints_InvalidDimensions1D` | ❌ |
| 98 | `TestValidatePoints_InvalidSingleValue` | ❌ |
| 99 | `TestGetTerminalSize_ReturnsValues` | N/A |
| 100 | `TestGetTerminalSize_CustomDefaults` | N/A |
| 101 | `TestGetTerminalSize_StandardDefaults` | N/A |
| 102 | `TestGetCutout_CenterRegion` | ❌ |
| 103 | `TestGetCutout_CornerRegion` | ❌ |
| 104 | `TestGetCutout_SinglePoint` | ❌ |
| 105 | `TestGetCutout_OutOfBounds` | ❌ |
| 106 | `TestGetCutout_LargeRegion` | ❌ |
| 107 | `TestGetCutout_InvalidPoints` | ❌ |

### `pkg/norfairgo/filter_test.go` (14 tests)
| # | Test | Ported? |
|---|------|---------|
| 108 | `TestFilterPyKalmanFilterFactory_Create` | ✅ |
| 109 | `TestFilterPyKalmanFilter_StaticObject` | ✅ |
| 110 | `TestFilterPyKalmanFilter_MovingObject` | ✅ |
| 111 | `TestOptimizedKalmanFilterFactory_Create` | ✅ |
| 112 | `TestOptimizedKalmanFilter_StaticObject` | ✅ |
| 113 | `TestOptimizedKalmanFilter_MovingObject` | ✅ |
| 114 | `TestNoFilterFactory_Create` | ✅ |
| 115 | `TestNoFilter_Predict` | ✅ |
| 116 | `TestNoFilter_Update` | ✅ |
| 117 | `TestFilterComparison_StaticObject` | ✅ |
| 118 | `TestFilterComparison_MovingObject` | ✅ |
| 119 | `TestFilterPyKalmanFilter_PartialMeasurement` | ✅ |
| 120 | `TestOptimizedKalmanFilter_PartialMeasurement` | ✅ |
| 121 | `TestFilters_MultiPoint` | ✅ |

### `pkg/norfairgo/camera_motion_test.go` (26 tests)
| # | Test | Ported? |
|---|------|---------|
| 122 | `TestTranslationTransformation_ForwardBackward` | ✅ |
| 123 | `TestTranslationTransformation_ZeroMovement` | ❌ |
| 124 | `TestTranslationTransformation_InvalidMovementVector` | ❌ |
| 125 | `TestTranslationTransformationGetter_SimpleModeFind` | ❌ |
| 126 | `TestTranslationTransformationGetter_Accumulation` | ❌ |
| 127 | `TestTranslationTransformationGetter_ReferenceUpdate` | ❌ |
| 128 | `TestTranslationTransformationGetter_SinglePoint` | ❌ |
| 129 | `TestTranslationTransformationGetter_MismatchedDimensions` | ❌ |
| 130 | `TestNilCoordinateTransformation` | ✅ |
| 131 | `TestHomographyTransformation_Identity` | ❌ |
| 132 | `TestHomographyTransformation_Translation` | ❌ |
| 133 | `TestHomographyTransformation_Scaling` | ❌ |
| 134 | `TestHomographyTransformation_ForwardBackward` | ❌ |
| 135 | `TestHomographyTransformation_DivisionByZero` | ❌ |
| 136 | `TestHomographyTransformation_InvalidMatrix` | ❌ |
| 137 | `TestHomographyTransformation_SingularMatrix` | ❌ |
| 138 | `TestHomographyTransformation_Rotation` | ❌ |
| 139 | `TestHomographyTransformationGetter_PerfectCorrespondence` | ❌ |
| 140 | `TestHomographyTransformationGetter_InsufficientPoints` | ❌ |
| 141 | `TestHomographyTransformationGetter_WithOutliers` | ❌ |
| 142 | `TestHomographyTransformationGetter_Accumulation` | ❌ |
| 143 | `TestMotionEstimator_Construction` | ❌ |
| 144 | `TestMotionEstimator_FirstFrameInitialization` | ❌ |
| 145 | `TestMotionEstimator_CloseResourcesCleanly` | ❌ |
| 146 | `TestMotionEstimator_ComputeTranslation_Small` | ❌ |
| 147 | `TestMotionEstimator_ComputeTranslation_Large` | ❌ |

### `pkg/norfairgo/video_test.go` (18 tests)
| # | Test | Ported? |
|---|------|---------|
| 148 | `TestVideo_InputValidation_BothNil` | ❌ |
| 149 | `TestVideo_InputValidation_BothSet` | ❌ |
| 150 | `TestVideo_InputValidation_FileNotFound` | ❌ |
| 151 | `TestVideo_GetCodecFourcc_AVI` | ❌ |
| 152 | `TestVideo_GetCodecFourcc_MP4` | ❌ |
| 153 | `TestVideo_GetCodecFourcc_CustomOverride` | ❌ |
| 154 | `TestVideo_GetCodecFourcc_UnsupportedExtension` | ❌ |
| 155 | `TestVideo_GetOutputFilePath_File` | ❌ |
| 156 | `TestVideo_GetOutputFilePath_DirectoryWithCamera` | ❌ |
| 157 | `TestVideo_GetOutputFilePath_DirectoryWithFile` | ❌ |
| 158 | `TestVideo_GetProgressDescription_Camera` | ❌ |
| 159 | `TestVideo_GetProgressDescription_File` | ❌ |
| 160 | `TestVideo_GetProgressDescription_WithLabel` | ❌ |
| 161 | `TestVideo_GetProgressDescription_Abbreviation` | ❌ |
| 162 | `TestVideoFromFrames_INIParsing` | ❌ |
| 163 | `TestVideoFromFrames_MissingINI` | ❌ |
| 164 | `TestVideoFromFrames_InvalidINI` | ❌ |
| 165 | `TestVideoFromFrames_VideoGeneration` | ❌ |

### `pkg/norfairgo/metrics_test.go` & `extended_metrics_test.go` (4 tests)
| # | Test | Ported? |
|---|------|---------|
| 166 | `TestEvalMotChallenge_Perfect` | ❌ |
| 167 | `TestEvalMotChallenge_MostlyLost` | ❌ |
| 168 | `TestEvalMotChallenge_Fragmented` | ❌ |
| 169 | `TestEvalMotChallenge_Mixed` | ❌ |

### `pkg/norfairgo/tracker_test.go` (7 tests)
| # | Test | Ported? |
|---|------|---------|
| 170 | `TestTracker_NewTracker` | ✅ |
| 171 | `TestTracker_InvalidInitializationDelay` | ✅ |
| 172 | `TestTracker_SimpleUpdate` | ✅ |
| 173 | `TestTracker_UpdateEmptyDetections` | ❌ |
| 174 | `TestDetection_Creation` | ✅ |
| 175 | `TestTrackedObject_Creation` | ✅ |
| 176 | `TestTracker_CameraMotion` | ✅ |

### `pkg/norfairgo/tracker_factory_test.go` (8 tests)
| # | Test | Ported? |
|---|------|---------|
| 177 | `TestTrackedObjectFactory_GetInitializingID` | ✅ |
| 178 | `TestTrackedObjectFactory_GetIDs` | ✅ |
| 179 | `TestTrackedObjectFactory_GlobalIDUniqueness` | ✅ |
| 180 | `TestTrackedObjectFactory_InitializingVsPermanentIDs` | ✅ |
| 181 | `TestTrackedObjectFactory_MixedSequence` | ✅ |
| 182 | `TestTrackedObjectFactory_ConcurrentInitializingIDs` | ✅ |
| 183 | `TestTrackedObjectFactory_ConcurrentPermanentIDs` | ✅ |
| 184 | `TestTrackedObjectFactory_ConcurrentMultipleFactories` | ✅ |

### `pkg/norfairgo/distances_test.go` (11 tests)
| # | Test | Ported? |
|---|------|---------|
| 185 | `TestFrobenius` | ✅ |
| 186 | `TestMeanManhattan` | ✅ |
| 187 | `TestMeanEuclidean` | ✅ |
| 188 | `TestIoU` | ✅ |
| 189 | `TestIoU_InvalidBbox` | ❌ |
| 190 | `TestScalarDistance` | ❌ |
| 191 | `TestVectorizedDistance` | ❌ |
| 192 | `TestScipyDistance` | ✅ |
| 193 | `TestKeypointVote` | ✅ |
| 194 | `TestNormalizedEuclidean` | ✅ |
| 195 | `TestGetDistanceByName` | ❌ |

### `pkg/norfairgo/benchmark_test.go` (6 benchmarks)
| # | Test | Ported? |
|---|------|---------|
| 196 | `BenchmarkTrackerUpdate_10Objects` | ❌ |
| 197 | `BenchmarkTrackerUpdate_50Objects` | ❌ |
| 198 | `BenchmarkTrackerUpdate_100Objects` | ❌ |
| 199 | `BenchmarkTrackerUpdate_100Objects_FilterPyKalman` | ❌ |
| 200 | `BenchmarkTrackerUpdate_100Objects_NoFilter` | ❌ |
| 201 | `BenchmarkTrackerUpdate_100Objects_IoU` | ❌ |

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

## Rust Tests (Current: ~186)

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

### `src/distances/functions.rs` (41 tests)
All ported from Python/Go

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

### `src/metrics/*.rs` (~4 tests)
Partial port from Go `metrics_test.go`

### `src/tracker.rs` (4 tests)
Partial port from Go/Python tracker tests

### `src/camera_motion/*.rs` (4 tests)
Partial port from Go `camera_motion_test.go`

---

## Implementation Priority

### Phase 1: Core Missing Tests (No New Dependencies)
1. `src/internal/scipy/distance.rs` - 8 missing cdist tests
2. `src/internal/scipy/optimize.rs` - 12 linear_sum_assignment tests
3. `src/internal/numpy/array.rs` - 7 missing linspace tests
4. `src/internal/motmetrics/iou.rs` - 10 missing IoU tests
5. `src/internal/motmetrics/accumulator.rs` - 17 accumulator tests
6. `src/matching.rs` - 2 helper function tests
7. `src/utils.rs` - 12 missing validation/cutout tests
8. `src/tracker.rs` - 4 missing tests
9. `src/camera_motion/transformations.rs` - 8 missing tests

### Phase 2: Wrapper Class Tests
1. `src/distances/scalar.rs` - ScalarDistance wrapper
2. `src/distances/vectorized.rs` - VectorizedDistance wrapper

### Phase 3: Metrics Tests
1. `src/metrics/evaluation.rs` - 4 eval_mot_challenge tests

### Phase 4: OpenCV-Dependent Tests (Behind Feature Flag)
1. Drawing tests (#202-323)
2. Video tests (#148-165)
3. Homography tests (#131-147)
