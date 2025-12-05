//! Vectorized distance wrapper for batch distance functions.

use super::traits::Distance;
use crate::internal::numpy::flatten_row_major;
use crate::{Detection, TrackedObject};
use nalgebra::DMatrix;

/// Vectorized distance function type.
///
/// Takes matrices of candidate and object points and returns a distance matrix.
pub type VectorizedDistanceFn = fn(&DMatrix<f64>, &DMatrix<f64>) -> DMatrix<f64>;

/// Wrapper for vectorized distance functions.
///
/// Takes a function that computes distances in batch (e.g., IoU)
/// and handles label filtering.
#[derive(Clone, Copy)]
pub struct VectorizedDistance {
    distance_fn: VectorizedDistanceFn,
}

impl VectorizedDistance {
    /// Create a new VectorizedDistance wrapper.
    pub fn new(distance_fn: VectorizedDistanceFn) -> Self {
        Self { distance_fn }
    }
}

impl Distance for VectorizedDistance {
    #[inline]
    fn get_distances(&self, objects: &[&TrackedObject], candidates: &[&Detection]) -> DMatrix<f64> {
        let n_candidates = candidates.len();
        let n_objects = objects.len();

        if n_candidates == 0 || n_objects == 0 {
            return DMatrix::zeros(n_candidates, n_objects);
        }

        // Build candidate and object matrices
        // For vectorized functions like IoU, we expect flattened bbox format [x1, y1, x2, y2]
        // IMPORTANT: Use row-major flattening to match Python/Go behavior
        let n_features = candidates[0].points.nrows() * candidates[0].points.ncols();
        let mut cand_matrix = DMatrix::zeros(n_candidates, n_features);
        let mut obj_matrix = DMatrix::zeros(n_objects, n_features);

        for (i, candidate) in candidates.iter().enumerate() {
            let flat = flatten_row_major(&candidate.points);
            for (j, &val) in flat.iter().enumerate() {
                cand_matrix[(i, j)] = val;
            }
        }

        for (i, object) in objects.iter().enumerate() {
            let flat = flatten_row_major(&object.estimate);
            for (j, &val) in flat.iter().enumerate() {
                obj_matrix[(i, j)] = val;
            }
        }

        // Compute base distances
        let mut result = (self.distance_fn)(&cand_matrix, &obj_matrix);

        // Apply label filtering (set to infinity if labels don't match)
        for (i, candidate) in candidates.iter().enumerate() {
            for (j, object) in objects.iter().enumerate() {
                if candidate.label.is_some()
                    && object.label.is_some()
                    && candidate.label != object.label
                {
                    result[(i, j)] = f64::INFINITY;
                }
            }
        }

        result
    }
}
