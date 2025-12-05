//! Detection struct for input to the tracker.

use crate::internal::numpy::validate_points;
use crate::{Error, Result};
use nalgebra::DMatrix;

/// A detection to be tracked.
///
/// Represents a detected object in a frame, with its position points
/// and optional metadata like scores, labels, and embeddings.
#[derive(Debug)]
pub struct Detection {
    /// Detection points (n_points x n_dims).
    /// For keypoints: [[x1, y1], [x2, y2], ...]
    /// For bounding boxes: [[x1, y1], [x2, y2]] (top-left, bottom-right)
    pub points: DMatrix<f64>,

    /// Optional per-point confidence scores.
    pub scores: Option<Vec<f64>>,

    /// Optional class label for multi-class tracking.
    pub label: Option<String>,

    /// Optional embedding vector for re-identification.
    pub embedding: Option<Vec<f64>>,

    /// Optional arbitrary user data.
    pub data: Option<Box<dyn std::any::Any + Send + Sync>>,

    /// Points in absolute coordinates (set by tracker when using camera motion).
    pub(crate) absolute_points: Option<DMatrix<f64>>,

    /// Frame age of this detection (set by tracker during update).
    /// This is the age of the TrackedObject when this detection was matched to it.
    pub age: Option<i32>,
}

impl Detection {
    /// Create a new detection with the given points.
    ///
    /// # Arguments
    /// * `points` - Detection points (n_points x n_dims)
    ///
    /// # Returns
    /// A new Detection instance
    pub fn new(points: DMatrix<f64>) -> Result<Self> {
        let validated = validate_points(&points)?;
        Ok(Self {
            points: validated.clone(),
            scores: None,
            label: None,
            embedding: None,
            data: None,
            absolute_points: Some(validated),
            age: None,
        })
    }

    /// Create a detection from a slice of points.
    ///
    /// # Arguments
    /// * `points` - Flat slice of points in row-major order
    /// * `n_points` - Number of points
    /// * `n_dims` - Number of dimensions per point
    pub fn from_slice(points: &[f64], n_points: usize, n_dims: usize) -> Result<Self> {
        if points.len() != n_points * n_dims {
            return Err(Error::InvalidDetection(format!(
                "Points slice length {} doesn't match {}x{}",
                points.len(),
                n_points,
                n_dims
            )));
        }

        let matrix = DMatrix::from_row_slice(n_points, n_dims, points);
        Self::new(matrix)
    }

    /// Create a detection with optional configuration.
    pub fn with_config(
        points: DMatrix<f64>,
        scores: Option<Vec<f64>>,
        label: Option<String>,
        embedding: Option<Vec<f64>>,
    ) -> Result<Self> {
        let validated = validate_points(&points)?;

        if let Some(ref s) = scores {
            if s.len() != validated.nrows() {
                return Err(Error::InvalidDetection(format!(
                    "Scores length {} doesn't match {} points",
                    s.len(),
                    validated.nrows()
                )));
            }
        }

        Ok(Self {
            points: validated.clone(),
            scores,
            label,
            embedding,
            data: None,
            absolute_points: Some(validated),
            age: None,
        })
    }

    /// Get the number of points in this detection.
    pub fn num_points(&self) -> usize {
        self.points.nrows()
    }

    /// Get the dimensionality of points (typically 2 for 2D tracking).
    pub fn num_dims(&self) -> usize {
        self.points.ncols()
    }

    /// Get absolute points (in world coordinates after camera motion compensation).
    pub fn get_absolute_points(&self) -> &DMatrix<f64> {
        self.absolute_points.as_ref().unwrap_or(&self.points)
    }

    /// Set absolute points (called by tracker during coordinate transformation).
    pub(crate) fn set_absolute_points(&mut self, points: DMatrix<f64>) {
        self.absolute_points = Some(points);
    }
}

impl Clone for Detection {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            scores: self.scores.clone(),
            label: self.label.clone(),
            embedding: self.embedding.clone(),
            data: None, // User data is not cloned
            absolute_points: self.absolute_points.clone(),
            age: self.age,
        }
    }
}

impl Default for Detection {
    fn default() -> Self {
        Self {
            points: DMatrix::zeros(1, 2),
            scores: None,
            label: None,
            embedding: None,
            data: None,
            absolute_points: None,
            age: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_detection_new() {
        let points = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let det = Detection::new(points).unwrap();

        assert_eq!(det.num_points(), 2);
        assert_eq!(det.num_dims(), 2);
        assert_relative_eq!(det.points[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_detection_from_slice() {
        let det = Detection::from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();

        assert_eq!(det.num_points(), 2);
        assert_eq!(det.num_dims(), 2);
    }

    #[test]
    fn test_detection_with_scores() {
        let points = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let det = Detection::with_config(points, Some(vec![0.9, 0.8]), None, None).unwrap();

        assert_eq!(det.scores.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_detection_with_label() {
        let points = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        let det = Detection::with_config(points, None, Some("person".to_string()), None).unwrap();

        assert_eq!(det.label.as_ref().unwrap(), "person");
    }
}
