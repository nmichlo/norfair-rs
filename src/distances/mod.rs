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
