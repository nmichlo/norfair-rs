//! Vectorized distance wrapper for batch distance functions.

use nalgebra::DMatrix;
use crate::{Detection, TrackedObject};
use super::traits::Distance;

/// Vectorized distance function type.
///
/// Takes matrices of candidate and object points and returns a distance matrix.
pub type VectorizedDistanceFn = fn(&DMatrix<f64>, &DMatrix<f64>) -> DMatrix<f64>;

/// Wrapper for vectorized distance functions.
///
/// Takes a function that computes distances in batch (e.g., IoU)
/// and handles label filtering.
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
    fn get_distances(
        &self,
        objects: &[&TrackedObject],
        candidates: &[&Detection],
    ) -> DMatrix<f64> {
        let n_candidates = candidates.len();
        let n_objects = objects.len();

        if n_candidates == 0 || n_objects == 0 {
            return DMatrix::zeros(n_candidates, n_objects);
        }

        // Build candidate and object matrices
        let candidate_rows: usize = candidates.iter().map(|c| c.points.nrows()).sum();
        let object_rows: usize = objects.iter().map(|o| o.estimate.nrows()).sum();
        let n_dims = candidates[0].points.ncols();

        // For vectorized functions like IoU, we expect flattened bbox format
        // Each detection/object contributes one row to the matrix
        let mut cand_matrix = DMatrix::zeros(n_candidates, n_dims * candidates[0].points.nrows());
        let mut obj_matrix = DMatrix::zeros(n_objects, n_dims * objects[0].estimate.nrows());

        for (i, candidate) in candidates.iter().enumerate() {
            let flat: Vec<f64> = candidate.points.iter().cloned().collect();
            for (j, &val) in flat.iter().enumerate() {
                if j < cand_matrix.ncols() {
                    cand_matrix[(i, j)] = val;
                }
            }
        }

        for (i, object) in objects.iter().enumerate() {
            let flat: Vec<f64> = object.estimate.iter().cloned().collect();
            for (j, &val) in flat.iter().enumerate() {
                if j < obj_matrix.ncols() {
                    obj_matrix[(i, j)] = val;
                }
            }
        }

        // Compute base distances
        let mut result = (self.distance_fn)(&cand_matrix, &obj_matrix);

        // Apply label filtering (set to infinity if labels don't match)
        for (i, candidate) in candidates.iter().enumerate() {
            for (j, object) in objects.iter().enumerate() {
                if candidate.label.is_some() && object.label.is_some() {
                    if candidate.label != object.label {
                        result[(i, j)] = f64::INFINITY;
                    }
                }
            }
        }

        result
    }
}
