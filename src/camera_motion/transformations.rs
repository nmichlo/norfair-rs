//! Coordinate transformation implementations.

use std::collections::HashMap;
use nalgebra::DMatrix;

/// Trait for transforming between relative and absolute coordinates.
///
/// This is used for camera motion compensation in tracking.
///
/// Detections' and tracked objects' coordinates can be interpreted in 2 references:
/// - Relative: their position on the current frame, (0, 0) is top left
/// - Absolute: their position in a fixed space, (0, 0) is the top left of the first frame
pub trait CoordinateTransformation: Send + Sync + std::fmt::Debug {
    /// Transform points from relative (camera frame) to absolute (world frame) coordinates.
    fn rel_to_abs(&self, points: &DMatrix<f64>) -> DMatrix<f64>;

    /// Transform points from absolute (world frame) to relative (camera frame) coordinates.
    fn abs_to_rel(&self, points: &DMatrix<f64>) -> DMatrix<f64>;

    /// Clone this transformation into a boxed trait object.
    fn clone_box(&self) -> Box<dyn CoordinateTransformation>;
}

impl Clone for Box<dyn CoordinateTransformation> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Trait for computing coordinate transformations between point correspondences.
pub trait TransformationGetter: Send + Sync {
    /// Compute the transformation between current and previous points.
    ///
    /// # Returns
    /// Tuple of (should_update_reference, transformation)
    fn call(&mut self, curr_pts: &DMatrix<f64>, prev_pts: &DMatrix<f64>) -> (bool, Option<Box<dyn CoordinateTransformation>>);
}

/// No-op transformation that returns points unchanged.
///
/// Used when camera motion is not being tracked.
#[derive(Debug, Clone, Default)]
pub struct NilCoordinateTransformation;

impl CoordinateTransformation for NilCoordinateTransformation {
    fn rel_to_abs(&self, points: &DMatrix<f64>) -> DMatrix<f64> {
        points.clone()
    }

    fn abs_to_rel(&self, points: &DMatrix<f64>) -> DMatrix<f64> {
        points.clone()
    }

    fn clone_box(&self) -> Box<dyn CoordinateTransformation> {
        Box::new(self.clone())
    }
}

/// Simple 2D translation transformation (camera pan/tilt without rotation/zoom).
#[derive(Debug, Clone)]
pub struct TranslationTransformation {
    /// Movement vector [dx, dy].
    pub movement_vector: [f64; 2],
}

impl TranslationTransformation {
    /// Create a new translation transformation with the given movement vector.
    pub fn new(movement_vector: [f64; 2]) -> Self {
        Self { movement_vector }
    }
}

impl CoordinateTransformation for TranslationTransformation {
    /// Convert absolute coordinates to relative by adding the movement vector.
    fn abs_to_rel(&self, points: &DMatrix<f64>) -> DMatrix<f64> {
        if points.ncols() != 2 {
            return points.clone();
        }

        let mut result = points.clone();
        for i in 0..result.nrows() {
            result[(i, 0)] += self.movement_vector[0];
            result[(i, 1)] += self.movement_vector[1];
        }
        result
    }

    /// Convert relative coordinates to absolute by subtracting the movement vector.
    fn rel_to_abs(&self, points: &DMatrix<f64>) -> DMatrix<f64> {
        if points.ncols() != 2 {
            return points.clone();
        }

        let mut result = points.clone();
        for i in 0..result.nrows() {
            result[(i, 0)] -= self.movement_vector[0];
            result[(i, 1)] -= self.movement_vector[1];
        }
        result
    }

    fn clone_box(&self) -> Box<dyn CoordinateTransformation> {
        Box::new(self.clone())
    }
}

/// Calculates translation transformation between points using optical flow mode.
///
/// The camera movement is calculated as the mode of optical flow between the previous
/// reference frame and the current. Comparing consecutive frames can make differences
/// too small to correctly estimate the translation, so the reference frame is kept fixed
/// as we progress through the video. Eventually, if the transformation can no longer
/// match enough points, the reference frame is updated.
pub struct TranslationTransformationGetter {
    /// Granularity for flow bucketing before calculating the mode.
    pub bin_size: f64,

    /// Minimum proportion of points that must be matched.
    pub proportion_points_used_threshold: f64,

    /// Accumulated transformation from the original reference frame.
    data: Option<[f64; 2]>,
}

impl TranslationTransformationGetter {
    /// Create a new translation transformation getter.
    pub fn new(bin_size: f64, proportion_points_used_threshold: f64) -> Self {
        Self {
            bin_size,
            proportion_points_used_threshold,
            data: None,
        }
    }
}

impl TransformationGetter for TranslationTransformationGetter {
    fn call(&mut self, curr_pts: &DMatrix<f64>, prev_pts: &DMatrix<f64>) -> (bool, Option<Box<dyn CoordinateTransformation>>) {
        let curr_rows = curr_pts.nrows();
        let prev_rows = prev_pts.nrows();

        if curr_rows != prev_rows || curr_pts.ncols() != 2 || prev_pts.ncols() != 2 {
            return (true, Some(Box::new(TranslationTransformation::new([0.0, 0.0]))));
        }

        // Step 1: Calculate flow = currPts - prevPts
        let mut flow: Vec<[f64; 2]> = Vec::with_capacity(curr_rows);
        for i in 0..curr_rows {
            flow.push([
                curr_pts[(i, 0)] - prev_pts[(i, 0)],
                curr_pts[(i, 1)] - prev_pts[(i, 1)],
            ]);
        }

        // Step 2: Bin the flow vectors (round to nearest bin_size)
        let binned_flow: Vec<[f64; 2]> = flow.iter().map(|f| {
            [
                (f[0] / self.bin_size).round() * self.bin_size,
                (f[1] / self.bin_size).round() * self.bin_size,
            ]
        }).collect();

        // Step 3: Find mode (most common flow vector)
        let mut flow_counts: HashMap<String, usize> = HashMap::new();
        let mut flow_vectors: HashMap<String, [f64; 2]> = HashMap::new();

        for f in &binned_flow {
            let key = format!("{:.10},{:.10}", f[0], f[1]);
            *flow_counts.entry(key.clone()).or_insert(0) += 1;
            flow_vectors.entry(key).or_insert(*f);
        }

        // Find flow with maximum count
        let mut max_key = String::new();
        let mut max_count = 0;
        for (key, &count) in &flow_counts {
            if count > max_count {
                max_count = count;
                max_key = key.clone();
            }
        }

        let mut flow_mode = flow_vectors.get(&max_key).copied().unwrap_or([0.0, 0.0]);

        // Step 4: Check proportion of points using the mode
        let proportion_points_used = max_count as f64 / curr_rows as f64;
        let update_prvs = proportion_points_used < self.proportion_points_used_threshold;

        // Step 5: Accumulate with previous transformation
        if let Some(prev_data) = self.data {
            flow_mode[0] += prev_data[0];
            flow_mode[1] += prev_data[1];
        }

        // Update accumulated data if reference frame should be updated
        if update_prvs {
            self.data = Some(flow_mode);
        }

        (update_prvs, Some(Box::new(TranslationTransformation::new(flow_mode))))
    }
}

/// Full perspective transformation using a 3x3 homography matrix.
///
/// Requires the `opencv` feature for creation from point correspondences.
#[cfg(feature = "opencv")]
#[derive(Debug, Clone)]
pub struct HomographyTransformation {
    /// 3x3 transformation matrix.
    pub homography_matrix: DMatrix<f64>,
    /// Pre-computed inverse for efficiency.
    pub inverse_homography_matrix: DMatrix<f64>,
}

#[cfg(feature = "opencv")]
impl HomographyTransformation {
    /// Create a new homography transformation with the given 3x3 matrix.
    pub fn new(homography_matrix: DMatrix<f64>) -> crate::Result<Self> {
        if homography_matrix.nrows() != 3 || homography_matrix.ncols() != 3 {
            return Err(crate::Error::TransformError(format!(
                "homography matrix must be 3x3, got {}x{}",
                homography_matrix.nrows(),
                homography_matrix.ncols()
            )));
        }

        // Compute inverse
        let inverse = homography_matrix.clone().try_inverse()
            .ok_or_else(|| crate::Error::TransformError("cannot invert homography matrix".to_string()))?;

        Ok(Self {
            homography_matrix,
            inverse_homography_matrix: inverse,
        })
    }

    /// Apply homography transformation to 2D points.
    fn transform_points(&self, points: &DMatrix<f64>, transform_matrix: &DMatrix<f64>) -> DMatrix<f64> {
        if points.ncols() != 2 {
            return points.clone();
        }

        let rows = points.nrows();
        let mut result = DMatrix::zeros(rows, 2);

        for i in 0..rows {
            let x = points[(i, 0)];
            let y = points[(i, 1)];

            // Apply homogeneous transformation: [x', y', w'] = H * [x, y, 1]^T
            let x_prime = transform_matrix[(0, 0)] * x + transform_matrix[(0, 1)] * y + transform_matrix[(0, 2)];
            let y_prime = transform_matrix[(1, 0)] * x + transform_matrix[(1, 1)] * y + transform_matrix[(1, 2)];
            let w_prime = transform_matrix[(2, 0)] * x + transform_matrix[(2, 1)] * y + transform_matrix[(2, 2)];

            // Perspective division
            let w = if w_prime == 0.0 { 0.0000001 } else { w_prime };
            result[(i, 0)] = x_prime / w;
            result[(i, 1)] = y_prime / w;
        }

        result
    }
}

#[cfg(feature = "opencv")]
impl CoordinateTransformation for HomographyTransformation {
    fn abs_to_rel(&self, points: &DMatrix<f64>) -> DMatrix<f64> {
        self.transform_points(points, &self.homography_matrix)
    }

    fn rel_to_abs(&self, points: &DMatrix<f64>) -> DMatrix<f64> {
        self.transform_points(points, &self.inverse_homography_matrix)
    }

    fn clone_box(&self) -> Box<dyn CoordinateTransformation> {
        Box::new(self.clone())
    }
}

/// Calculates homography transformation using RANSAC.
///
/// Requires the `opencv` feature.
#[cfg(feature = "opencv")]
pub struct HomographyTransformationGetter {
    /// Maximum allowed reprojection error for RANSAC inliers.
    pub ransac_reproj_threshold: f64,
    /// Maximum RANSAC iterations.
    pub max_iters: i32,
    /// RANSAC confidence level (0-1).
    pub confidence: f64,
    /// Minimum proportion of points that must be matched.
    pub proportion_points_used_threshold: f64,
    /// Accumulated homography from original reference frame.
    data: Option<DMatrix<f64>>,
}

#[cfg(feature = "opencv")]
impl HomographyTransformationGetter {
    /// Create a new homography transformation getter.
    pub fn new(
        ransac_reproj_threshold: f64,
        max_iters: i32,
        confidence: f64,
        proportion_points_used_threshold: f64,
    ) -> Self {
        Self {
            ransac_reproj_threshold,
            max_iters,
            confidence,
            proportion_points_used_threshold,
            data: None,
        }
    }
}

#[cfg(feature = "opencv")]
impl TransformationGetter for HomographyTransformationGetter {
    fn call(&mut self, _curr_pts: &DMatrix<f64>, _prev_pts: &DMatrix<f64>) -> (bool, Option<Box<dyn CoordinateTransformation>>) {
        // TODO: Implement using OpenCV's findHomography
        // This requires opencv crate integration
        (true, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nil_transformation() {
        let transform = NilCoordinateTransformation;
        let points = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        let abs = transform.rel_to_abs(&points);
        let rel = transform.abs_to_rel(&points);

        assert_eq!(abs, points);
        assert_eq!(rel, points);
    }

    #[test]
    fn test_translation_transformation() {
        let transform = TranslationTransformation::new([10.0, 20.0]);
        let points = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        // abs_to_rel adds the movement vector
        let rel = transform.abs_to_rel(&points);
        assert_relative_eq!(rel[(0, 0)], 11.0, epsilon = 1e-10);
        assert_relative_eq!(rel[(0, 1)], 22.0, epsilon = 1e-10);

        // rel_to_abs subtracts the movement vector
        let abs = transform.rel_to_abs(&points);
        assert_relative_eq!(abs[(0, 0)], -9.0, epsilon = 1e-10);
        assert_relative_eq!(abs[(0, 1)], -18.0, epsilon = 1e-10);
    }

    #[test]
    fn test_translation_roundtrip() {
        let transform = TranslationTransformation::new([10.0, 20.0]);
        let points = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        let intermediate = transform.abs_to_rel(&points);
        let roundtrip = transform.rel_to_abs(&intermediate);

        for i in 0..points.nrows() {
            for j in 0..points.ncols() {
                assert_relative_eq!(roundtrip[(i, j)], points[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_translation_getter() {
        let mut getter = TranslationTransformationGetter::new(1.0, 0.5);

        // All points have same flow vector [5, 10]
        let prev_pts = DMatrix::from_row_slice(3, 2, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let curr_pts = DMatrix::from_row_slice(3, 2, &[5.0, 10.0, 6.0, 11.0, 7.0, 12.0]);

        let (update_prvs, transform) = getter.call(&curr_pts, &prev_pts);

        // High proportion matched, should not update
        assert!(!update_prvs);
        assert!(transform.is_some());
    }
}
