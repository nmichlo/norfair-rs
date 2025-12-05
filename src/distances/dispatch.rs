//! Enum-based distance dispatch for static (non-virtual) function calls.
//!
//! This module provides `DistanceFunction`, an enum that wraps all supported
//! distance types and dispatches without vtable lookups, improving performance
//! for hot-path code.

#[cfg(feature = "python")]
use std::sync::Arc;

use super::functions::{frobenius, iou, mean_euclidean, mean_manhattan};
use super::scalar::ScalarDistance;
use super::scipy_wrapper::ScipyDistance;
use super::traits::Distance;
use super::vectorized::VectorizedDistance;
use crate::{Detection, TrackedObject};
use nalgebra::DMatrix;

/// Custom distance function type for Python callbacks.
///
/// Uses `Arc` to allow cloning while sharing the underlying function.
#[cfg(feature = "python")]
pub type CustomDistanceFn =
    Arc<dyn Fn(&[&TrackedObject], &[&Detection]) -> DMatrix<f64> + Send + Sync>;

/// Wrapper for custom distance functions (e.g., Python callables).
#[cfg(feature = "python")]
#[derive(Clone)]
pub struct CustomDistance {
    func: CustomDistanceFn,
}

#[cfg(feature = "python")]
impl CustomDistance {
    /// Create a new custom distance from a function.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&[&TrackedObject], &[&Detection]) -> DMatrix<f64> + Send + Sync + 'static,
    {
        Self { func: Arc::new(f) }
    }

    /// Get distances between objects and candidates.
    #[inline]
    pub fn get_distances(
        &self,
        objects: &[&TrackedObject],
        candidates: &[&Detection],
    ) -> DMatrix<f64> {
        (self.func)(objects, candidates)
    }

    /// Get distances between two sets of TrackedObjects (for ReID matching).
    ///
    /// This creates temporary Detections from the candidate TrackedObjects.
    /// The underlying Python callback will receive (Detection, TrackedObject).
    ///
    /// Note: For Python callables that expect (TrackedObject, TrackedObject),
    /// this won't work correctly. A separate reid-specific callback type would be needed.
    #[inline]
    pub fn get_distances_objects(
        &self,
        objects: &[&TrackedObject],
        candidates: &[&TrackedObject],
    ) -> DMatrix<f64> {
        // Create temporary detections from candidate estimates
        let temp_detections: Vec<Detection> = candidates
            .iter()
            .map(|obj| Detection {
                points: obj.estimate.clone(),
                scores: None,
                label: obj.label.clone(),
                embedding: None,
                data: None,
                absolute_points: Some(obj.estimate.clone()),
                age: Some(obj.age),
            })
            .collect();

        let det_refs: Vec<&Detection> = temp_detections.iter().collect();
        (self.func)(objects, &det_refs)
    }
}

/// Enum-based distance function for static dispatch.
///
/// This avoids `Box<dyn Distance>` vtable overhead by using an enum
/// with inline implementations. Use `distance_function_by_name()` to
/// create instances.
#[derive(Clone)]
pub enum DistanceFunction {
    // Scalar distance functions
    Frobenius(ScalarDistance),
    MeanEuclidean(ScalarDistance),
    MeanManhattan(ScalarDistance),

    // Vectorized distance functions
    Iou(VectorizedDistance),

    // Scipy-style distance functions
    ScipyEuclidean(ScipyDistance),
    ScipySqeuclidean(ScipyDistance),
    ScipyManhattan(ScipyDistance),
    ScipyCosine(ScipyDistance),
    ScipyChebyshev(ScipyDistance),

    /// Custom distance function (used for Python callables).
    /// Only available with the "python" feature.
    #[cfg(feature = "python")]
    Custom(CustomDistance),
}

impl DistanceFunction {
    /// Get distances between objects and candidates.
    #[inline(always)]
    pub fn get_distances(
        &self,
        objects: &[&TrackedObject],
        candidates: &[&Detection],
    ) -> DMatrix<f64> {
        match self {
            // Scalar functions
            DistanceFunction::Frobenius(d) => d.get_distances(objects, candidates),
            DistanceFunction::MeanEuclidean(d) => d.get_distances(objects, candidates),
            DistanceFunction::MeanManhattan(d) => d.get_distances(objects, candidates),

            // Vectorized functions
            DistanceFunction::Iou(d) => d.get_distances(objects, candidates),

            // Scipy functions
            DistanceFunction::ScipyEuclidean(d) => d.get_distances(objects, candidates),
            DistanceFunction::ScipySqeuclidean(d) => d.get_distances(objects, candidates),
            DistanceFunction::ScipyManhattan(d) => d.get_distances(objects, candidates),
            DistanceFunction::ScipyCosine(d) => d.get_distances(objects, candidates),
            DistanceFunction::ScipyChebyshev(d) => d.get_distances(objects, candidates),

            // Custom distance function (Python callables)
            #[cfg(feature = "python")]
            DistanceFunction::Custom(d) => d.get_distances(objects, candidates),
        }
    }

    /// Get distances between two sets of TrackedObjects (for ReID matching).
    ///
    /// For built-in distance functions, creates temporary Detections from candidate estimates.
    /// For custom Python callables, this requires the reid_distance_function to accept
    /// (TrackedObject, TrackedObject) -> float (not Detection, TrackedObject).
    #[inline(always)]
    pub fn get_distances_objects(
        &self,
        objects: &[&TrackedObject],
        candidates: &[&TrackedObject],
    ) -> DMatrix<f64> {
        // For built-in functions, create temporary detections from candidate estimates
        // and use the standard distance computation
        let temp_detections: Vec<Detection> = candidates
            .iter()
            .map(|obj| Detection {
                points: obj.estimate.clone(),
                scores: None,
                label: obj.label.clone(),
                embedding: None,
                data: None,
                absolute_points: Some(obj.estimate.clone()),
                age: Some(obj.age),
            })
            .collect();

        let det_refs: Vec<&Detection> = temp_detections.iter().collect();

        match self {
            // For Custom (Python callback), we need special handling
            // The callback expects (TrackedObject, TrackedObject), not (Detection, TrackedObject)
            #[cfg(feature = "python")]
            DistanceFunction::Custom(d) => d.get_distances_objects(objects, candidates),

            // For all built-in functions, use the standard detection-based distance
            _ => self.get_distances(objects, &det_refs),
        }
    }
}

/// Create a DistanceFunction enum by name (static dispatch version).
///
/// This is the preferred way to create distance functions for performance-critical code.
///
/// # Panics
/// Panics if the distance name is not recognized.
pub fn distance_function_by_name(name: &str) -> DistanceFunction {
    match name {
        // Scalar functions
        "frobenius" => DistanceFunction::Frobenius(ScalarDistance::new(frobenius)),
        "mean_euclidean" => DistanceFunction::MeanEuclidean(ScalarDistance::new(mean_euclidean)),
        "mean_manhattan" => DistanceFunction::MeanManhattan(ScalarDistance::new(mean_manhattan)),

        // Vectorized functions
        "iou" => DistanceFunction::Iou(VectorizedDistance::new(iou)),

        // Scipy functions
        "euclidean" => DistanceFunction::ScipyEuclidean(ScipyDistance::new("euclidean")),
        "sqeuclidean" => DistanceFunction::ScipySqeuclidean(ScipyDistance::new("sqeuclidean")),
        "manhattan" | "cityblock" => {
            DistanceFunction::ScipyManhattan(ScipyDistance::new("manhattan"))
        }
        "cosine" => DistanceFunction::ScipyCosine(ScipyDistance::new("cosine")),
        "chebyshev" => DistanceFunction::ScipyChebyshev(ScipyDistance::new("chebyshev")),

        _ => panic!("Unknown distance function: {}", name),
    }
}

/// Create a DistanceFunction enum by name, returning a Result instead of panicking.
///
/// This is useful for error handling when the distance name comes from user input.
pub fn try_distance_function_by_name(name: &str) -> Result<DistanceFunction, String> {
    match name {
        // Scalar functions
        "frobenius" => Ok(DistanceFunction::Frobenius(ScalarDistance::new(frobenius))),
        "mean_euclidean" => Ok(DistanceFunction::MeanEuclidean(ScalarDistance::new(mean_euclidean))),
        "mean_manhattan" => Ok(DistanceFunction::MeanManhattan(ScalarDistance::new(mean_manhattan))),

        // Vectorized functions
        "iou" => Ok(DistanceFunction::Iou(VectorizedDistance::new(iou))),

        // Scipy functions
        "euclidean" => Ok(DistanceFunction::ScipyEuclidean(ScipyDistance::new("euclidean"))),
        "sqeuclidean" => Ok(DistanceFunction::ScipySqeuclidean(ScipyDistance::new("sqeuclidean"))),
        "manhattan" | "cityblock" => Ok(DistanceFunction::ScipyManhattan(ScipyDistance::new("manhattan"))),
        "cosine" => Ok(DistanceFunction::ScipyCosine(ScipyDistance::new("cosine"))),
        "chebyshev" => Ok(DistanceFunction::ScipyChebyshev(ScipyDistance::new("chebyshev"))),

        _ => Err(format!("Unknown distance function: {}. Supported: frobenius, mean_euclidean, mean_manhattan, iou, euclidean, sqeuclidean, manhattan, cityblock, cosine, chebyshev", name)),
    }
}

// Implement the Distance trait for DistanceFunction so it can be used interchangeably
impl Distance for DistanceFunction {
    #[inline(always)]
    fn get_distances(&self, objects: &[&TrackedObject], candidates: &[&Detection]) -> DMatrix<f64> {
        // Delegate to the inherent method
        DistanceFunction::get_distances(self, objects, candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_mock_detection(points: &[f64], rows: usize, cols: usize) -> Detection {
        Detection {
            points: DMatrix::from_row_slice(rows, cols, points),
            scores: None,
            label: None,
            embedding: None,
            data: None,
            absolute_points: None,
            age: None,
        }
    }

    fn create_mock_tracked_object(estimate: &[f64], rows: usize, cols: usize) -> TrackedObject {
        let estimate_matrix = DMatrix::from_row_slice(rows, cols, estimate);
        TrackedObject {
            id: Some(0),
            global_id: 0,
            initializing_id: None,
            age: 0,
            hit_counter: 1,
            point_hit_counter: vec![1; rows],
            last_detection: None,
            last_distance: None,
            current_min_distance: None,
            past_detections: std::collections::VecDeque::new(),
            label: None,
            reid_hit_counter: None,
            estimate: estimate_matrix.clone(),
            estimate_velocity: DMatrix::zeros(rows, cols),
            is_initializing: false,
            detected_at_least_once_points: vec![true; rows],
            filter: crate::filter::FilterEnum::None(crate::filter::NoFilter::new(&estimate_matrix)),
            initial_period: 1,
            num_points: rows,
            dim_points: cols,
            last_coord_transform: None,
        }
    }

    #[test]
    fn test_distance_function_frobenius() {
        let distance = distance_function_by_name("frobenius");
        let det = create_mock_detection(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let matrix = distance.get_distances(&[&obj], &[&det]);
        assert!((matrix[(0, 0)] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_function_iou() {
        let distance = distance_function_by_name("iou");
        let det = create_mock_detection(&[0.0, 0.0, 1.0, 1.0], 1, 4);
        let obj = create_mock_tracked_object(&[0.0, 0.0, 1.0, 1.0], 1, 4);
        let matrix = distance.get_distances(&[&obj], &[&det]);
        assert!((matrix[(0, 0)] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_function_euclidean() {
        let distance = distance_function_by_name("euclidean");
        let det = create_mock_detection(&[1.0, 2.0], 1, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0], 1, 2);
        let matrix = distance.get_distances(&[&obj], &[&det]);
        assert!((matrix[(0, 0)] - 0.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Unknown distance function")]
    fn test_distance_function_invalid() {
        distance_function_by_name("invalid_distance");
    }

    // ===== CustomDistance Tests (Python feature only) =====

    #[cfg(feature = "python")]
    #[test]
    fn test_custom_distance_basic() {
        use std::sync::Arc;

        // Create a simple custom distance function that returns euclidean distance
        let custom = CustomDistance::new(|objects, candidates| {
            let n_cands = candidates.len();
            let n_objs = objects.len();
            let mut matrix = DMatrix::zeros(n_cands, n_objs);

            for (c, cand) in candidates.iter().enumerate() {
                for (o, obj) in objects.iter().enumerate() {
                    // Simple euclidean distance between first points
                    let det_point = cand.points.row(0);
                    let obj_point = obj.estimate.row(0);
                    let diff: f64 = det_point
                        .iter()
                        .zip(obj_point.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    matrix[(c, o)] = diff.sqrt();
                }
            }
            matrix
        });

        let det = create_mock_detection(&[1.0, 2.0], 1, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0], 1, 2);

        let matrix = custom.get_distances(&[&obj], &[&det]);
        assert!(
            (matrix[(0, 0)] - 0.0).abs() < 1e-6,
            "Perfect match should have distance 0"
        );
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_custom_distance_nonzero() {
        // Custom distance that returns a fixed value
        let custom = CustomDistance::new(|objects, candidates| {
            let n_cands = candidates.len();
            let n_objs = objects.len();
            let mut matrix = DMatrix::zeros(n_cands, n_objs);
            for c in 0..n_cands {
                for o in 0..n_objs {
                    matrix[(c, o)] = 42.0; // Fixed distance
                }
            }
            matrix
        });

        let det = create_mock_detection(&[1.0, 2.0], 1, 2);
        let obj = create_mock_tracked_object(&[100.0, 200.0], 1, 2);

        let matrix = custom.get_distances(&[&obj], &[&det]);
        assert!(
            (matrix[(0, 0)] - 42.0).abs() < 1e-6,
            "Should return fixed value 42"
        );
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_custom_distance_multiple_objects_and_detections() {
        // Custom distance that returns row + col index
        let custom = CustomDistance::new(|objects, candidates| {
            let n_cands = candidates.len();
            let n_objs = objects.len();
            let mut matrix = DMatrix::zeros(n_cands, n_objs);
            for c in 0..n_cands {
                for o in 0..n_objs {
                    matrix[(c, o)] = (c + o) as f64;
                }
            }
            matrix
        });

        let det1 = create_mock_detection(&[1.0, 1.0], 1, 2);
        let det2 = create_mock_detection(&[2.0, 2.0], 1, 2);
        let obj1 = create_mock_tracked_object(&[10.0, 10.0], 1, 2);
        let obj2 = create_mock_tracked_object(&[20.0, 20.0], 1, 2);

        let matrix = custom.get_distances(&[&obj1, &obj2], &[&det1, &det2]);

        // Matrix should be 2x2 (2 candidates x 2 objects)
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);

        // Check values: matrix[(c, o)] = c + o
        assert!((matrix[(0, 0)] - 0.0).abs() < 1e-6); // c=0, o=0
        assert!((matrix[(0, 1)] - 1.0).abs() < 1e-6); // c=0, o=1
        assert!((matrix[(1, 0)] - 1.0).abs() < 1e-6); // c=1, o=0
        assert!((matrix[(1, 1)] - 2.0).abs() < 1e-6); // c=1, o=1
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_distance_function_custom_variant() {
        // Test DistanceFunction::Custom variant works through the enum dispatch
        let custom = CustomDistance::new(|_objects, _candidates| DMatrix::from_element(1, 1, 5.5));

        let distance = DistanceFunction::Custom(custom);

        let det = create_mock_detection(&[1.0, 2.0], 1, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0], 1, 2);

        let matrix = distance.get_distances(&[&obj], &[&det]);
        assert!(
            (matrix[(0, 0)] - 5.5).abs() < 1e-6,
            "Custom distance should return 5.5"
        );
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_custom_distance_clone() {
        // Test that CustomDistance can be cloned (via Arc)
        let custom = CustomDistance::new(|_objects, _candidates| DMatrix::from_element(1, 1, 7.0));

        let custom_clone = custom.clone();

        let det = create_mock_detection(&[1.0, 2.0], 1, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0], 1, 2);

        // Both should return the same value
        let matrix1 = custom.get_distances(&[&obj], &[&det]);
        let matrix2 = custom_clone.get_distances(&[&obj], &[&det]);

        assert!((matrix1[(0, 0)] - 7.0).abs() < 1e-6);
        assert!((matrix2[(0, 0)] - 7.0).abs() < 1e-6);
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_distance_function_custom_clone() {
        // Test that DistanceFunction::Custom can be cloned
        let custom = CustomDistance::new(|_objects, _candidates| DMatrix::from_element(1, 1, 3.14));

        let distance = DistanceFunction::Custom(custom);
        let distance_clone = distance.clone();

        let det = create_mock_detection(&[1.0, 2.0], 1, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0], 1, 2);

        let matrix1 = distance.get_distances(&[&obj], &[&det]);
        let matrix2 = distance_clone.get_distances(&[&obj], &[&det]);

        assert!((matrix1[(0, 0)] - 3.14).abs() < 1e-6);
        assert!((matrix2[(0, 0)] - 3.14).abs() < 1e-6);
    }
}
