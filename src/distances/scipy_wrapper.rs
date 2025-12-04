//! SciPy-style distance wrapper using cdist.

use nalgebra::DMatrix;
use crate::{Detection, TrackedObject};
use crate::internal::scipy::cdist;
use super::traits::Distance;

/// Wrapper for scipy-style distance metrics.
///
/// Uses cdist to compute pairwise distances with a specified metric.
pub struct ScipyDistance {
    metric: String,
}

impl ScipyDistance {
    /// Create a new ScipyDistance with the specified metric.
    ///
    /// Supported metrics: "euclidean", "manhattan", "cosine", "chebyshev", etc.
    pub fn new(metric: &str) -> Self {
        Self {
            metric: metric.to_string(),
        }
    }
}

impl Distance for ScipyDistance {
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

        // Flatten points for each candidate and object
        let n_features = candidates[0].points.nrows() * candidates[0].points.ncols();

        let mut cand_matrix = DMatrix::zeros(n_candidates, n_features);
        let mut obj_matrix = DMatrix::zeros(n_objects, n_features);

        for (i, candidate) in candidates.iter().enumerate() {
            let flat: Vec<f64> = candidate.points.iter().cloned().collect();
            for (j, &val) in flat.iter().enumerate() {
                if j < n_features {
                    cand_matrix[(i, j)] = val;
                }
            }
        }

        for (i, object) in objects.iter().enumerate() {
            let flat: Vec<f64> = object.estimate.iter().cloned().collect();
            for (j, &val) in flat.iter().enumerate() {
                if j < n_features {
                    obj_matrix[(i, j)] = val;
                }
            }
        }

        // Compute distances using cdist
        let mut result = cdist(&cand_matrix, &obj_matrix, &self.metric);

        // Apply label filtering
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
