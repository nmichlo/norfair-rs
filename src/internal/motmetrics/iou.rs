//! IoU (Intersection over Union) computation for MOT evaluation.
//!
//! Ported from py-motmetrics IoU distance computation.
//! License: MIT (Christoph Heindl, Jack Valmadre)
#![allow(dead_code)]

use nalgebra::DMatrix;

/// Compute IoU distance between two bounding boxes.
///
/// Returns 1.0 - IoU (distance in range [0, 1]):
/// - 0.0 = perfect overlap (IoU = 1.0)
/// - 1.0 = no overlap (IoU = 0.0)
///
/// # Arguments
/// * `box1` - Bounding box [x_min, y_min, x_max, y_max]
/// * `box2` - Bounding box [x_min, y_min, x_max, y_max]
///
/// # Panics
/// Panics if boxes don't have 4 elements or have invalid coordinates.
pub fn iou_distance(box1: &[f64], box2: &[f64]) -> f64 {
    // Validate input
    if box1.len() != 4 || box2.len() != 4 {
        panic!(
            "boxes must have 4 elements [x_min, y_min, x_max, y_max], got {} and {}",
            box1.len(),
            box2.len()
        );
    }

    // Validate box1 coordinates
    if box1[2] <= box1[0] || box1[3] <= box1[1] {
        panic!(
            "invalid box1: x_max ({:.2}) <= x_min ({:.2}) or y_max ({:.2}) <= y_min ({:.2})",
            box1[2], box1[0], box1[3], box1[1]
        );
    }

    // Validate box2 coordinates
    if box2[2] <= box2[0] || box2[3] <= box2[1] {
        panic!(
            "invalid box2: x_max ({:.2}) <= x_min ({:.2}) or y_max ({:.2}) <= y_min ({:.2})",
            box2[2], box2[0], box2[3], box2[1]
        );
    }

    // Compute intersection rectangle
    let x_min_inter = box1[0].max(box2[0]);
    let y_min_inter = box1[1].max(box2[1]);
    let x_max_inter = box1[2].min(box2[2]);
    let y_max_inter = box1[3].min(box2[3]);

    // Compute intersection area
    let intersection = if x_max_inter < x_min_inter || y_max_inter < y_min_inter {
        0.0 // No overlap
    } else {
        (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    };

    // Compute union area
    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let union = area1 + area2 - intersection;

    // Edge case: zero union (both boxes have zero area)
    if union == 0.0 {
        return 1.0; // Maximum distance
    }

    // Compute IoU and convert to distance
    let iou = intersection / union;
    1.0 - iou
}

/// Compute pairwise IoU distances for all GT × prediction pairs.
///
/// # Arguments
/// * `gt_bboxes` - Ground truth bounding boxes, each [x_min, y_min, x_max, y_max]
/// * `pred_bboxes` - Predicted bounding boxes, same format
///
/// # Returns
/// Distance matrix [numGT][numPred] where each element is IoU distance (1.0 - IoU)
pub fn compute_iou_distance_matrix(
    gt_bboxes: &[Vec<f64>],
    pred_bboxes: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let num_gt = gt_bboxes.len();
    let num_pred = pred_bboxes.len();

    let mut matrix = vec![vec![0.0; num_pred]; num_gt];
    for i in 0..num_gt {
        for j in 0..num_pred {
            matrix[i][j] = iou_distance(&gt_bboxes[i], &pred_bboxes[j]);
        }
    }

    matrix
}

/// Compute IoU matrix between two sets of bounding boxes.
///
/// # Arguments
/// * `boxes_a` - First set of boxes, shape (n, 4), format [x1, y1, x2, y2]
/// * `boxes_b` - Second set of boxes, shape (m, 4), format [x1, y1, x2, y2]
///
/// # Returns
/// IoU matrix of shape (n, m)
pub fn iou_matrix(boxes_a: &DMatrix<f64>, boxes_b: &DMatrix<f64>) -> DMatrix<f64> {
    let n = boxes_a.nrows();
    let m = boxes_b.nrows();

    if n == 0 || m == 0 {
        return DMatrix::zeros(n, m);
    }

    let mut result = DMatrix::zeros(n, m);

    for i in 0..n {
        let a_x1 = boxes_a[(i, 0)];
        let a_y1 = boxes_a[(i, 1)];
        let a_x2 = boxes_a[(i, 2)];
        let a_y2 = boxes_a[(i, 3)];
        let a_area = (a_x2 - a_x1) * (a_y2 - a_y1);

        for j in 0..m {
            let b_x1 = boxes_b[(j, 0)];
            let b_y1 = boxes_b[(j, 1)];
            let b_x2 = boxes_b[(j, 2)];
            let b_y2 = boxes_b[(j, 3)];
            let b_area = (b_x2 - b_x1) * (b_y2 - b_y1);

            // Intersection
            let inter_x1 = a_x1.max(b_x1);
            let inter_y1 = a_y1.max(b_y1);
            let inter_x2 = a_x2.min(b_x2);
            let inter_y2 = a_y2.min(b_y2);

            let inter_w = (inter_x2 - inter_x1).max(0.0);
            let inter_h = (inter_y2 - inter_y1).max(0.0);
            let inter_area = inter_w * inter_h;

            // Union
            let union_area = a_area + b_area - inter_area;

            // IoU
            result[(i, j)] = if union_area > 0.0 {
                inter_area / union_area
            } else {
                0.0
            };
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ===== iou_matrix tests (IoU similarity) =====

    #[test]
    fn test_iou_perfect_overlap() {
        let boxes = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 10.0, 10.0]);
        let result = iou_matrix(&boxes, &boxes);
        assert_relative_eq!(result[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_no_overlap() {
        let boxes_a = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 10.0, 10.0]);
        let boxes_b = DMatrix::from_row_slice(1, 4, &[20.0, 20.0, 30.0, 30.0]);
        let result = iou_matrix(&boxes_a, &boxes_b);
        assert_relative_eq!(result[(0, 0)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let boxes_a = DMatrix::from_row_slice(1, 4, &[0.0, 0.0, 10.0, 10.0]);
        let boxes_b = DMatrix::from_row_slice(1, 4, &[5.0, 5.0, 15.0, 15.0]);
        let result = iou_matrix(&boxes_a, &boxes_b);
        // Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        assert_relative_eq!(result[(0, 0)], 25.0 / 175.0, epsilon = 1e-10);
    }

    // ===== iou_distance tests (IoU distance = 1 - IoU) =====

    #[test]
    fn test_iou_distance_perfect_overlap() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [0.0, 0.0, 10.0, 10.0];
        let distance = iou_distance(&box1, &box2);
        assert_relative_eq!(distance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_distance_no_overlap() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [20.0, 20.0, 30.0, 30.0];
        let distance = iou_distance(&box1, &box2);
        assert_relative_eq!(distance, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_distance_partial_overlap() {
        // Two 10x10 boxes with 5x10 overlap
        // Area1 = 100, Area2 = 100, Intersection = 50, Union = 150
        // IoU = 50/150 = 1/3, Distance = 1 - 1/3 = 2/3
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [5.0, 0.0, 15.0, 10.0];
        let distance = iou_distance(&box1, &box2);
        let expected = 1.0 - (1.0 / 3.0); // 2/3
        assert_relative_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_distance_contained_box() {
        // Small box inside large box
        // Intersection = 25, Union = 100
        // IoU = 25/100 = 0.25, Distance = 0.75
        let box1 = [0.0, 0.0, 10.0, 10.0]; // Area 100
        let box2 = [2.5, 2.5, 7.5, 7.5]; // Area 25, fully contained
        let distance = iou_distance(&box1, &box2);
        let expected = 1.0 - 0.25; // 0.75
        assert_relative_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_distance_adjacent_boxes() {
        // Two boxes touching at edge (no overlap)
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [10.0, 0.0, 20.0, 10.0]; // Touches at x=10
        let distance = iou_distance(&box1, &box2);
        assert_relative_eq!(distance, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_distance_small_overlap() {
        // Very small overlap region
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [9.0, 9.0, 19.0, 19.0];
        // Intersection = 1x1 = 1, Union = 100 + 100 - 1 = 199
        // IoU = 1/199, Distance ≈ 0.995
        let distance = iou_distance(&box1, &box2);
        let expected = 1.0 - (1.0 / 199.0);
        assert_relative_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_iou_distance_floating_point() {
        let box1 = [0.5, 0.5, 10.5, 10.5];
        let box2 = [5.5, 0.5, 15.5, 10.5];
        // Same as test_iou_distance_partial_overlap but with 0.5 offset
        let distance = iou_distance(&box1, &box2);
        let expected = 1.0 - (1.0 / 3.0);
        assert_relative_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "invalid box1")]
    fn test_iou_distance_invalid_box1() {
        let box1 = [10.0, 10.0, 0.0, 0.0]; // x_max < x_min
        let box2 = [0.0, 0.0, 10.0, 10.0];
        iou_distance(&box1, &box2);
    }

    #[test]
    #[should_panic(expected = "invalid box2")]
    fn test_iou_distance_invalid_box2() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [0.0, 10.0, 10.0, 0.0]; // y_max < y_min
        iou_distance(&box1, &box2);
    }

    #[test]
    #[should_panic(expected = "boxes must have 4 elements")]
    fn test_iou_distance_wrong_length() {
        let box1 = [0.0, 0.0, 10.0]; // Only 3 elements
        let box2 = [0.0, 0.0, 10.0, 10.0];
        iou_distance(&box1, &box2);
    }

    // ===== compute_iou_distance_matrix tests =====

    #[test]
    fn test_compute_iou_matrix() {
        let gt_bboxes = vec![
            vec![0.0, 0.0, 10.0, 10.0],   // GT 0
            vec![20.0, 20.0, 30.0, 30.0], // GT 1
        ];
        let pred_bboxes = vec![
            vec![0.0, 0.0, 10.0, 10.0],   // Perfect match with GT 0
            vec![25.0, 25.0, 35.0, 35.0], // Overlaps with GT 1
            vec![50.0, 50.0, 60.0, 60.0], // No overlap with any GT
        ];

        let matrix = compute_iou_distance_matrix(&gt_bboxes, &pred_bboxes);

        // Verify dimensions
        assert_eq!(matrix.len(), 2);
        for (i, row) in matrix.iter().enumerate() {
            assert_eq!(row.len(), 3, "Row {} should have 3 columns", i);
        }

        // Verify specific distances
        assert_relative_eq!(matrix[0][0], 0.0, epsilon = 1e-10); // GT0-Pred0: perfect match
        assert_relative_eq!(matrix[0][2], 1.0, epsilon = 1e-10); // GT0-Pred2: no overlap
        assert_relative_eq!(matrix[1][2], 1.0, epsilon = 1e-10); // GT1-Pred2: no overlap

        // GT1-Pred1 should have some overlap
        assert!(
            matrix[1][1] > 0.0 && matrix[1][1] < 1.0,
            "GT1-Pred1 should have partial overlap distance in (0, 1), got {}",
            matrix[1][1]
        );
    }

    #[test]
    fn test_compute_iou_matrix_empty() {
        // Empty GT boxes
        let gt_bboxes: Vec<Vec<f64>> = vec![];
        let pred_bboxes = vec![vec![0.0, 0.0, 10.0, 10.0]];
        let matrix = compute_iou_distance_matrix(&gt_bboxes, &pred_bboxes);
        assert_eq!(matrix.len(), 0);

        // Empty pred boxes
        let gt_bboxes = vec![vec![0.0, 0.0, 10.0, 10.0]];
        let pred_bboxes: Vec<Vec<f64>> = vec![];
        let matrix = compute_iou_distance_matrix(&gt_bboxes, &pred_bboxes);
        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0].len(), 0);
    }

    #[test]
    fn test_compute_iou_matrix_large_set() {
        // Generate 10 GT boxes and 15 pred boxes
        let gt_bboxes: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let x = i as f64 * 15.0;
                vec![x, 0.0, x + 10.0, 10.0]
            })
            .collect();

        let pred_bboxes: Vec<Vec<f64>> = (0..15)
            .map(|i| {
                let x = i as f64 * 10.0;
                vec![x, 0.0, x + 10.0, 10.0]
            })
            .collect();

        let matrix = compute_iou_distance_matrix(&gt_bboxes, &pred_bboxes);

        // Verify dimensions
        assert_eq!(matrix.len(), 10);
        for (i, row) in matrix.iter().enumerate() {
            assert_eq!(row.len(), 15, "Row {} should have 15 columns", i);
        }

        // Verify all distances are in valid range [0, 1]
        for (i, row) in matrix.iter().enumerate() {
            for (j, &dist) in row.iter().enumerate() {
                assert!(
                    dist >= 0.0 && dist <= 1.0,
                    "Distance[{}][{}] = {} outside valid range [0, 1]",
                    i,
                    j,
                    dist
                );
            }
        }
    }
}
