//! Enum-based distance dispatch for static (non-virtual) function calls.
//!
//! This module provides `DistanceFunction`, an enum that wraps all supported
//! distance types and dispatches without vtable lookups, improving performance
//! for hot-path code.

use nalgebra::DMatrix;
use crate::{Detection, TrackedObject};
use super::traits::Distance;
use super::scalar::ScalarDistance;
use super::vectorized::VectorizedDistance;
use super::scipy_wrapper::ScipyDistance;
use super::functions::{frobenius, mean_euclidean, mean_manhattan, iou};

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
        "manhattan" | "cityblock" => DistanceFunction::ScipyManhattan(ScipyDistance::new("manhattan")),
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
    fn get_distances(
        &self,
        objects: &[&TrackedObject],
        candidates: &[&Detection],
    ) -> DMatrix<f64> {
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
}
