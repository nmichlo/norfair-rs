//! Distance trait definition.

use nalgebra::DMatrix;
use crate::{Detection, TrackedObject};

/// Trait for distance functions used in object matching.
///
/// Distance functions compute a matrix of distances between candidate
/// detections and tracked objects. Lower distances indicate better matches.
pub trait Distance: Send + Sync {
    /// Compute distances between objects and candidate detections.
    ///
    /// # Arguments
    /// * `objects` - Slice of tracked objects
    /// * `candidates` - Slice of candidate detections
    ///
    /// # Returns
    /// Distance matrix of shape (n_candidates, n_objects).
    /// Entry (i, j) is the distance between candidate i and object j.
    fn get_distances(
        &self,
        objects: &[&TrackedObject],
        candidates: &[&Detection],
    ) -> DMatrix<f64>;
}
