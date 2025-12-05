//! Scalar distance wrapper for per-pair distance functions.

use super::traits::Distance;
use crate::{Detection, TrackedObject};
use nalgebra::DMatrix;

/// Scalar distance function type.
pub type ScalarDistanceFn = fn(&Detection, &TrackedObject) -> f64;

/// Wrapper for scalar distance functions.
///
/// Takes a function that computes distance between a single detection
/// and tracked object pair, and produces a full distance matrix.
#[derive(Clone, Copy)]
pub struct ScalarDistance {
    distance_fn: ScalarDistanceFn,
}

impl ScalarDistance {
    /// Create a new ScalarDistance wrapper.
    pub fn new(distance_fn: ScalarDistanceFn) -> Self {
        Self { distance_fn }
    }
}

impl Distance for ScalarDistance {
    #[inline]
    fn get_distances(&self, objects: &[&TrackedObject], candidates: &[&Detection]) -> DMatrix<f64> {
        let n_candidates = candidates.len();
        let n_objects = objects.len();

        let mut result = DMatrix::from_element(n_candidates, n_objects, f64::INFINITY);

        for (i, candidate) in candidates.iter().enumerate() {
            for (j, object) in objects.iter().enumerate() {
                // Skip if labels don't match
                if candidate.label.is_some()
                    && object.label.is_some()
                    && candidate.label != object.label
                {
                    continue;
                }

                result[(i, j)] = (self.distance_fn)(candidate, object);
            }
        }

        result
    }
}
