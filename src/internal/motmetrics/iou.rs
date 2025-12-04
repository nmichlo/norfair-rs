//! IoU (Intersection over Union) computation for MOT evaluation.

use nalgebra::DMatrix;

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
}
