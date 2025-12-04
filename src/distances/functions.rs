//! Built-in distance functions.

use nalgebra::DMatrix;
use crate::{Detection, TrackedObject};

/// Frobenius norm distance between detection and tracked object points.
///
/// Computes sqrt(sum((det.points - obj.estimate)^2))
pub fn frobenius(detection: &Detection, object: &TrackedObject) -> f64 {
    let diff = &detection.points - &object.estimate;
    diff.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Mean Euclidean distance per point.
///
/// Computes the average L2 distance across all points.
pub fn mean_euclidean(detection: &Detection, object: &TrackedObject) -> f64 {
    let n_points = detection.points.nrows();
    if n_points == 0 {
        return f64::INFINITY;
    }

    let mut total_dist = 0.0;
    for i in 0..n_points {
        let det_row = detection.points.row(i);
        let obj_row = object.estimate.row(i);
        let dist: f64 = det_row
            .iter()
            .zip(obj_row.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        total_dist += dist;
    }

    total_dist / n_points as f64
}

/// Mean Manhattan distance per point.
///
/// Computes the average L1 distance across all points.
pub fn mean_manhattan(detection: &Detection, object: &TrackedObject) -> f64 {
    let n_points = detection.points.nrows();
    if n_points == 0 {
        return f64::INFINITY;
    }

    let mut total_dist = 0.0;
    for i in 0..n_points {
        let det_row = detection.points.row(i);
        let obj_row = object.estimate.row(i);
        let dist: f64 = det_row
            .iter()
            .zip(obj_row.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        total_dist += dist;
    }

    total_dist / n_points as f64
}

/// IoU (Intersection over Union) distance for bounding boxes.
///
/// Expects boxes in format [[x1, y1, x2, y2]] (min and max corners).
/// Returns 1 - IoU, so lower is better (perfect match = 0).
pub fn iou(candidates: &DMatrix<f64>, objects: &DMatrix<f64>) -> DMatrix<f64> {
    let n_cand = candidates.nrows();
    let n_obj = objects.nrows();

    assert!(
        candidates.ncols() >= 4,
        "IoU requires at least 4 columns (x1, y1, x2, y2), got {}",
        candidates.ncols()
    );
    assert!(
        objects.ncols() >= 4,
        "IoU requires at least 4 columns (x1, y1, x2, y2), got {}",
        objects.ncols()
    );

    let mut result = DMatrix::zeros(n_cand, n_obj);

    for i in 0..n_cand {
        let c_x1 = candidates[(i, 0)];
        let c_y1 = candidates[(i, 1)];
        let c_x2 = candidates[(i, 2)];
        let c_y2 = candidates[(i, 3)];
        let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);

        for j in 0..n_obj {
            let o_x1 = objects[(j, 0)];
            let o_y1 = objects[(j, 1)];
            let o_x2 = objects[(j, 2)];
            let o_y2 = objects[(j, 3)];
            let o_area = (o_x2 - o_x1) * (o_y2 - o_y1);

            // Intersection
            let inter_x1 = c_x1.max(o_x1);
            let inter_y1 = c_y1.max(o_y1);
            let inter_x2 = c_x2.min(o_x2);
            let inter_y2 = c_y2.min(o_y2);

            let inter_w = (inter_x2 - inter_x1).max(0.0);
            let inter_h = (inter_y2 - inter_y1).max(0.0);
            let inter_area = inter_w * inter_h;

            // Union
            let union_area = c_area + o_area - inter_area;

            // IoU distance (1 - IoU)
            result[(i, j)] = if union_area > 0.0 {
                1.0 - inter_area / union_area
            } else {
                1.0
            };
        }
    }

    result
}

/// Create a keypoints voting distance function.
///
/// # Arguments
/// * `keypoint_distance_threshold` - Maximum distance for a keypoint to be considered matching
/// * `detection_threshold` - Minimum score for a keypoint to be considered
pub fn create_keypoints_voting_distance(
    keypoint_distance_threshold: f64,
    detection_threshold: f64,
) -> impl Fn(&Detection, &TrackedObject) -> f64 {
    move |detection: &Detection, object: &TrackedObject| {
        let n_points = detection.points.nrows();
        let mut matches = 0;
        let mut total_valid = 0;

        for i in 0..n_points {
            // Check scores
            let det_score = detection.scores.as_ref().map(|s| s[i]).unwrap_or(1.0);
            let obj_score = object
                .last_detection
                .as_ref()
                .and_then(|d| d.scores.as_ref().map(|s| s[i]))
                .unwrap_or(1.0);

            if det_score <= detection_threshold || obj_score <= detection_threshold {
                continue;
            }

            total_valid += 1;

            // Compute distance for this point
            let det_row = detection.points.row(i);
            let obj_row = object.estimate.row(i);
            let dist: f64 = det_row
                .iter()
                .zip(obj_row.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if dist < keypoint_distance_threshold {
                matches += 1;
            }
        }

        if total_valid == 0 {
            return 1.0;
        }

        1.0 - (matches as f64 / (total_valid + 1) as f64)
    }
}

/// Create a normalized mean Euclidean distance function.
///
/// # Arguments
/// * `height` - Image height for normalization
/// * `width` - Image width for normalization
pub fn create_normalized_mean_euclidean_distance(
    height: f64,
    width: f64,
) -> impl Fn(&Detection, &TrackedObject) -> f64 {
    move |detection: &Detection, object: &TrackedObject| {
        let n_points = detection.points.nrows();
        if n_points == 0 {
            return f64::INFINITY;
        }

        let mut total_dist = 0.0;
        for i in 0..n_points {
            let det_row = detection.points.row(i);
            let obj_row = object.estimate.row(i);

            // Normalize x by width, y by height
            let mut sq_sum = 0.0;
            for (j, (a, b)) in det_row.iter().zip(obj_row.iter()).enumerate() {
                let norm = if j % 2 == 0 { width } else { height };
                let diff = (a - b) / norm;
                sq_sum += diff * diff;
            }
            total_dist += sq_sum.sqrt();
        }

        total_dist / n_points as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_detection(points: &[f64], n_rows: usize, n_cols: usize) -> Detection {
        Detection {
            points: DMatrix::from_row_slice(n_rows, n_cols, points),
            scores: None,
            label: None,
            embedding: None,
            data: None,
            absolute_points: None,
        }
    }

    fn make_object(points: &[f64], n_rows: usize, n_cols: usize) -> TrackedObject {
        TrackedObject {
            id: None,
            global_id: 0,
            initializing_id: None,
            estimate: DMatrix::from_row_slice(n_rows, n_cols, points),
            label: None,
            last_detection: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_frobenius_perfect_match() {
        let det = make_detection(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let obj = make_object(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_relative_eq!(frobenius(&det, &obj), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_frobenius_distance() {
        let det = make_detection(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let obj = make_object(&[2.0, 3.0, 4.0, 5.0], 2, 2);
        // sqrt(1+1+1+1) = sqrt(4) = 2.0
        assert_relative_eq!(frobenius(&det, &obj), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_euclidean_perfect_match() {
        let det = make_detection(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let obj = make_object(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_relative_eq!(mean_euclidean(&det, &obj), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_euclidean_distance() {
        let det = make_detection(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let obj = make_object(&[2.0, 3.0, 4.0, 5.0], 2, 2);
        // Each point: sqrt(1+1) = sqrt(2)
        // Mean: sqrt(2)
        assert_relative_eq!(mean_euclidean(&det, &obj), 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_iou_perfect_match() {
        let cand = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 1.0, 1.0]);
        let obj = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 1.0, 1.0]);
        let result = iou(&cand, &obj);
        assert_relative_eq!(result[(0, 0)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_no_overlap() {
        let cand = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 1.0, 1.0]);
        let obj = DMatrix::from_row_slice(1, 4, &[1.0, 1.0, 2.0, 2.0]);
        let result = iou(&cand, &obj);
        assert_relative_eq!(result[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let cand = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 2.0, 2.0]);
        let obj = DMatrix::from_row_slice(1, 4, &[1.0, 1.0, 3.0, 3.0]);
        let result = iou(&cand, &obj);
        // Intersection: 1x1 = 1, Union: 4+4-1 = 7
        assert_relative_eq!(result[(0, 0)], 1.0 - 1.0 / 7.0, epsilon = 1e-10);
    }
}
