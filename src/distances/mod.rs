//! Distance functions for matching detections to tracked objects.
//!
//! This module provides:
//! - `Distance` trait for all distance implementations
//! - `ScalarDistance` - wrapper for per-pair distance functions
//! - `VectorizedDistance` - wrapper for batch distance functions
//! - `ScipyDistance` - wrapper for scipy-style cdist metrics
//! - Built-in distance functions (frobenius, mean_euclidean, iou, etc.)

mod traits;
mod scalar;
mod vectorized;
mod scipy_wrapper;
mod functions;

pub use traits::Distance;
pub use scalar::ScalarDistance;
pub use vectorized::VectorizedDistance;
pub use scipy_wrapper::ScipyDistance;
pub use functions::*;

use crate::{Error, Result};

/// Get a distance function by name.
///
/// Supported names:
/// - "euclidean", "manhattan", "cosine", "chebyshev" - scipy metrics
/// - "frobenius" - Frobenius norm of difference
/// - "mean_euclidean" - Mean L2 distance per point
/// - "mean_manhattan" - Mean L1 distance per point
/// - "iou" - Intersection over Union for bounding boxes
pub fn distance_by_name(name: &str) -> Box<dyn Distance> {
    match name {
        // Scipy-style metrics
        "euclidean" | "sqeuclidean" | "manhattan" | "cityblock" | "cosine" | "chebyshev" => {
            Box::new(ScipyDistance::new(name))
        }
        // Scalar functions
        "frobenius" => Box::new(ScalarDistance::new(frobenius)),
        "mean_euclidean" => Box::new(ScalarDistance::new(mean_euclidean)),
        "mean_manhattan" => Box::new(ScalarDistance::new(mean_manhattan)),
        // Vectorized functions
        "iou" => Box::new(VectorizedDistance::new(iou)),
        _ => panic!("Unknown distance function: {}", name),
    }
}

/// Get a distance function by name, returning a Result instead of panicking.
pub fn try_distance_by_name(name: &str) -> Result<Box<dyn Distance>> {
    match name {
        "euclidean" | "sqeuclidean" | "manhattan" | "cityblock" | "cosine" | "chebyshev" => {
            Ok(Box::new(ScipyDistance::new(name)))
        }
        "frobenius" => Ok(Box::new(ScalarDistance::new(frobenius))),
        "mean_euclidean" => Ok(Box::new(ScalarDistance::new(mean_euclidean))),
        "mean_manhattan" => Ok(Box::new(ScalarDistance::new(mean_manhattan))),
        "iou" => Ok(Box::new(VectorizedDistance::new(iou))),
        _ => Err(Error::UnknownDistance(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Detection, TrackedObject};
    use nalgebra::DMatrix;

    fn create_mock_detection(points: &[f64], rows: usize, cols: usize) -> Detection {
        Detection {
            points: DMatrix::from_row_slice(rows, cols, points),
            scores: None,
            label: None,
            embedding: None,
            data: None,
            absolute_points: None,
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
            past_detections: std::collections::VecDeque::new(),
            label: None,
            reid_hit_counter: None,
            estimate: estimate_matrix.clone(),
            estimate_velocity: DMatrix::zeros(rows, cols),
            is_initializing: false,
            filter: Box::new(crate::filter::NoFilter::new(&estimate_matrix)),
            num_points: rows,
            dim_points: cols,
            last_coord_transform: None,
        }
    }

    // ===== ScalarDistance Wrapper Tests =====

    /// Ported from Go: TestScalarDistance
    #[test]
    fn test_scalar_distance_wrapper() {
        // Test the ScalarDistance wrapper with frobenius
        let distance = ScalarDistance::new(frobenius);

        let det = create_mock_detection(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0, 3.0, 4.0], 2, 2);

        let detections = vec![&det];
        let objects = vec![&obj];

        let matrix = distance.get_distances(&objects, &detections);

        assert_eq!(matrix.nrows(), 1, "Expected 1 row");
        assert_eq!(matrix.ncols(), 1, "Expected 1 column");
        assert!((matrix[(0, 0)] - 0.0).abs() < 1e-6, "frobenius distance should be 0");
    }

    // ===== VectorizedDistance Wrapper Tests =====

    /// Ported from Go: TestVectorizedDistance
    #[test]
    fn test_vectorized_distance_wrapper() {
        // Test the VectorizedDistance wrapper with IoU
        let distance = VectorizedDistance::new(iou);

        // Create bboxes: [x1, y1, x2, y2] format
        let det = create_mock_detection(&[0.0, 0.0, 1.0, 1.0], 1, 4);
        let obj = create_mock_tracked_object(&[0.0, 0.0, 1.0, 1.0], 1, 4);

        let detections = vec![&det];
        let objects = vec![&obj];

        let matrix = distance.get_distances(&objects, &detections);

        assert_eq!(matrix.nrows(), 1, "Expected 1 row");
        assert_eq!(matrix.ncols(), 1, "Expected 1 column");
        assert!((matrix[(0, 0)] - 0.0).abs() < 1e-6, "IoU distance should be 0 for perfect match");
    }

    // ===== ScipyDistance Wrapper Tests =====

    /// Ported from Go: TestScipyDistance
    #[test]
    fn test_scipy_distance_wrapper() {
        // Test ScipyDistance with euclidean metric
        let distance = ScipyDistance::new("euclidean");

        // det = [[1, 2], [3, 4]] flattened to [1, 2, 3, 4]
        // obj = [[1, 2], [4, 4]] flattened to [1, 2, 4, 4]
        // euclidean distance = sqrt((1-1)^2 + (2-2)^2 + (3-4)^2 + (4-4)^2) = sqrt(1) = 1.0
        let det = create_mock_detection(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0, 4.0, 4.0], 2, 2);

        let detections = vec![&det];
        let objects = vec![&obj];

        let matrix = distance.get_distances(&objects, &detections);

        assert_eq!(matrix.nrows(), 1, "Expected 1 row");
        assert_eq!(matrix.ncols(), 1, "Expected 1 column");
        assert!((matrix[(0, 0)] - 1.0).abs() < 1e-6, "euclidean distance should be 1.0");
    }

    // ===== distance_by_name Tests =====

    /// Ported from Go: TestGetDistanceByName (frobenius)
    #[test]
    fn test_distance_by_name_frobenius() {
        let distance = distance_by_name("frobenius");

        // Verify it works by computing a distance
        let det = create_mock_detection(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let matrix = distance.get_distances(&[&obj], &[&det]);
        assert!((matrix[(0, 0)] - 0.0).abs() < 1e-6);
    }

    /// Ported from Go: TestGetDistanceByName (iou)
    #[test]
    fn test_distance_by_name_iou() {
        let distance = distance_by_name("iou");

        // Verify it works with bbox format
        let det = create_mock_detection(&[0.0, 0.0, 1.0, 1.0], 1, 4);
        let obj = create_mock_tracked_object(&[0.0, 0.0, 1.0, 1.0], 1, 4);
        let matrix = distance.get_distances(&[&obj], &[&det]);
        assert!((matrix[(0, 0)] - 0.0).abs() < 1e-6);
    }

    /// Ported from Go: TestGetDistanceByName (euclidean)
    #[test]
    fn test_distance_by_name_euclidean() {
        let distance = distance_by_name("euclidean");

        // Verify it works
        let det = create_mock_detection(&[1.0, 2.0], 1, 2);
        let obj = create_mock_tracked_object(&[1.0, 2.0], 1, 2);
        let matrix = distance.get_distances(&[&obj], &[&det]);
        assert!((matrix[(0, 0)] - 0.0).abs() < 1e-6);
    }

    /// Ported from Go: TestGetDistanceByName (invalid_distance)
    #[test]
    fn test_try_distance_by_name_invalid() {
        let result = try_distance_by_name("invalid_distance");
        assert!(result.is_err(), "Expected error for invalid distance name");
    }

    #[test]
    #[should_panic(expected = "Unknown distance function")]
    fn test_distance_by_name_panics_on_invalid() {
        distance_by_name("invalid_distance");
    }
}
