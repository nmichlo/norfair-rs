//! Detection-to-object matching algorithms.

use nalgebra::DMatrix;
use crate::{Error, Result};

/// Check if a matrix contains NaN values.
pub fn has_nan(matrix: &DMatrix<f64>) -> bool {
    matrix.iter().any(|&x| x.is_nan())
}

/// Validate a distance matrix (no NaN values allowed).
pub fn validate_distance_matrix(matrix: &DMatrix<f64>) -> Result<()> {
    if has_nan(matrix) {
        return Err(Error::DistanceError("Distance matrix contains NaN values".to_string()));
    }
    Ok(())
}

/// Match detections to tracked objects using greedy minimum-distance matching.
///
/// # Arguments
/// * `distance_matrix` - Distance matrix (n_detections x n_objects)
/// * `threshold` - Maximum distance for a valid match
///
/// # Returns
/// Tuple of (matched_det_indices, matched_obj_indices) where entry i indicates
/// the matched pair. Unmatched detections/objects are not included.
pub fn match_detections_and_objects(
    distance_matrix: &DMatrix<f64>,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let n_detections = distance_matrix.nrows();
    let n_objects = distance_matrix.ncols();

    if n_detections == 0 || n_objects == 0 {
        return (Vec::new(), Vec::new());
    }

    // Collect all valid (distance, det_idx, obj_idx) pairs
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n_detections {
        for j in 0..n_objects {
            let dist = distance_matrix[(i, j)];
            if dist.is_finite() && dist <= threshold {
                pairs.push((dist, i, j));
            }
        }
    }

    // Sort by distance (ascending)
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Greedy matching with Vec<bool> (faster than HashSet)
    let mut used_dets = vec![false; n_detections];
    let mut used_objs = vec![false; n_objects];

    let mut matched_dets = Vec::new();
    let mut matched_objs = Vec::new();

    for (_dist, det_idx, obj_idx) in pairs {
        if used_dets[det_idx] || used_objs[obj_idx] {
            continue;
        }

        matched_dets.push(det_idx);
        matched_objs.push(obj_idx);
        used_dets[det_idx] = true;
        used_objs[obj_idx] = true;
    }

    (matched_dets, matched_objs)
}

/// Get unmatched indices from a match result.
pub fn get_unmatched(total: usize, matched: &[usize]) -> Vec<usize> {
    let mut is_matched = vec![false; total];
    for &idx in matched {
        is_matched[idx] = true;
    }
    (0..total).filter(|&i| !is_matched[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Test Perfect Matches =====

    #[test]
    fn test_perfect_matches() {
        // All distances below threshold - all should match
        let matrix = DMatrix::from_row_slice(3, 3, &[
            0.5, 0.9, 0.8,
            0.9, 0.3, 0.7,
            0.8, 0.7, 0.4,
        ]);
        let (dets, objs) = match_detections_and_objects(&matrix, 1.0);

        // Should match all 3 pairs
        assert_eq!(dets.len(), 3);
        assert_eq!(objs.len(), 3);

        // Greedy order: [1,1]=0.3, [2,2]=0.4, [0,0]=0.5
        assert_eq!(dets, vec![1, 2, 0]);
        assert_eq!(objs, vec![1, 2, 0]);
    }

    // ===== Test Threshold Filtering =====

    #[test]
    fn test_threshold_filtering() {
        // Some distances above threshold, some below
        let matrix = DMatrix::from_row_slice(3, 3, &[
            0.5, 2.0, 3.0,
            2.5, 0.8, 2.0,
            3.0, 3.0, 0.3,
        ]);
        let (dets, objs) = match_detections_and_objects(&matrix, 1.5);

        // Greedy order: [2,2]=0.3, [0,0]=0.5, [1,1]=0.8
        assert_eq!(dets, vec![2, 0, 1]);
        assert_eq!(objs, vec![2, 0, 1]);
    }

    #[test]
    fn test_all_above_threshold() {
        // All distances above threshold - no matches
        let matrix = DMatrix::from_row_slice(2, 2, &[
            5.0, 6.0,
            7.0, 8.0,
        ]);
        let (dets, objs) = match_detections_and_objects(&matrix, 3.0);

        assert!(dets.is_empty());
        assert!(objs.is_empty());
    }

    // ===== Test Empty/Minimal Inputs =====

    #[test]
    fn test_match_empty() {
        let matrix = DMatrix::zeros(0, 0);
        let (dets, objs) = match_detections_and_objects(&matrix, 1.0);
        assert!(dets.is_empty());
        assert!(objs.is_empty());
    }

    #[test]
    fn test_single_element_no_match() {
        // 1x1 matrix with distance above threshold
        let matrix = DMatrix::from_row_slice(1, 1, &[5.0]);
        let (dets, objs) = match_detections_and_objects(&matrix, 3.0);
        assert!(dets.is_empty());
        assert!(objs.is_empty());
    }

    #[test]
    fn test_single_element_match() {
        // 1x1 matrix with distance below threshold
        let matrix = DMatrix::from_row_slice(1, 1, &[0.5]);
        let (dets, objs) = match_detections_and_objects(&matrix, 1.0);
        assert_eq!(dets, vec![0]);
        assert_eq!(objs, vec![0]);
    }

    // ===== Test Greedy Behavior =====

    #[test]
    fn test_greedy_behavior() {
        // Matrix where greedy picks diagonal matches
        let matrix = DMatrix::from_row_slice(2, 2, &[
            1.0, 2.0,
            2.0, 1.0,
        ]);
        let (dets, objs) = match_detections_and_objects(&matrix, 3.0);

        // Should match both
        assert_eq!(dets.len(), 2);
        assert_eq!(objs.len(), 2);

        // Verify one-to-one mapping (no duplicates)
        use std::collections::HashSet;
        let det_set: HashSet<_> = dets.iter().collect();
        let obj_set: HashSet<_> = objs.iter().collect();
        assert_eq!(det_set.len(), 2);
        assert_eq!(obj_set.len(), 2);
    }

    // ===== Test One-to-One Constraint =====

    #[test]
    fn test_one_to_one_constraint() {
        // Multiple candidates close to same object
        let matrix = DMatrix::from_row_slice(3, 2, &[
            0.5, 3.0,  // Cand 0 closest to Obj 0
            0.6, 3.5,  // Cand 1 also close to Obj 0
            0.7, 2.0,  // Cand 2 closest to Obj 1
        ]);
        let (dets, objs) = match_detections_and_objects(&matrix, 4.0);

        // Should have 2 matches (one per object)
        assert_eq!(dets.len(), 2);

        // Greedy picks: [0,0]=0.5 first, then [2,1]=2.0
        assert_eq!(dets, vec![0, 2]);
        assert_eq!(objs, vec![0, 1]);
    }

    // ===== Test Asymmetric Matrices =====

    #[test]
    fn test_more_detections_than_objects() {
        // 5 detections, 3 objects
        let matrix = DMatrix::from_row_slice(5, 3, &[
            0.5, 2.0, 3.0,
            0.8, 0.4, 2.5,
            1.2, 1.5, 0.3,
            2.0, 2.5, 1.8,
            3.0, 3.5, 2.2,
        ]);
        let (dets, objs) = match_detections_and_objects(&matrix, 2.0);

        // Should match at most 3 (limited by objects)
        assert!(dets.len() <= 3);

        // Greedy picks: [2,2]=0.3, [1,1]=0.4, [0,0]=0.5
        assert_eq!(dets, vec![2, 1, 0]);
        assert_eq!(objs, vec![2, 1, 0]);
    }

    #[test]
    fn test_more_objects_than_detections() {
        // 2 detections, 4 objects
        let matrix = DMatrix::from_row_slice(2, 4, &[
            0.5, 2.0, 1.5, 3.0,
            1.8, 0.6, 2.5, 2.2,
        ]);
        let (dets, objs) = match_detections_and_objects(&matrix, 2.0);

        // Should match at most 2 (limited by detections)
        assert!(dets.len() <= 2);

        // Greedy picks: [0,0]=0.5, [1,1]=0.6
        assert_eq!(dets, vec![0, 1]);
        assert_eq!(objs, vec![0, 1]);
    }

    // ===== Test NaN Detection =====

    #[test]
    fn test_nan_detection() {
        let matrix = DMatrix::from_row_slice(2, 2, &[
            0.5, f64::NAN,
            1.0, 0.8,
        ]);

        assert!(has_nan(&matrix));
        assert!(validate_distance_matrix(&matrix).is_err());
    }

    #[test]
    fn test_no_nan() {
        let matrix = DMatrix::from_row_slice(2, 2, &[
            0.5, 1.0,
            1.5, 0.8,
        ]);

        assert!(!has_nan(&matrix));
        assert!(validate_distance_matrix(&matrix).is_ok());
    }

    // ===== Test Inf Handling =====

    #[test]
    fn test_inf_handling() {
        // Matrix with Inf values (from label mismatches in distance functions)
        let matrix = DMatrix::from_row_slice(3, 3, &[
            0.5, f64::INFINITY, f64::INFINITY,
            f64::INFINITY, 0.8, f64::INFINITY,
            f64::INFINITY, f64::INFINITY, 0.3,
        ]);
        let (dets, objs) = match_detections_and_objects(&matrix, 1.0);

        // Should match the 3 finite values below threshold
        // Greedy order: [2,2]=0.3, [0,0]=0.5, [1,1]=0.8
        assert_eq!(dets, vec![2, 0, 1]);
        assert_eq!(objs, vec![2, 0, 1]);

        // Verify Inf doesn't cause NaN error
        assert!(validate_distance_matrix(&matrix).is_ok());
    }

    // ===== Test get_unmatched =====

    #[test]
    fn test_get_unmatched() {
        let unmatched = get_unmatched(5, &[1, 3]);
        assert_eq!(unmatched, vec![0, 2, 4]);
    }

    #[test]
    fn test_get_unmatched_none() {
        let unmatched = get_unmatched(3, &[0, 1, 2]);
        assert!(unmatched.is_empty());
    }

    #[test]
    fn test_get_unmatched_all() {
        let unmatched = get_unmatched(3, &[]);
        assert_eq!(unmatched, vec![0, 1, 2]);
    }

    // ===== Test minimum finding behavior (equivalent to Go's TestArgMin/TestMinMatrix) =====

    #[test]
    fn test_finds_minimum_value() {
        // Matrix with minimum at a specific position
        let matrix = DMatrix::from_row_slice(3, 3, &[
            5.0, 3.0, 7.0,
            2.0, 9.0, 4.0,
            6.0, 1.0, 8.0,  // Minimum is at [2,1] = 1.0
        ]);

        let (dets, objs) = match_detections_and_objects(&matrix, 10.0);

        // Should find and match the minimum (1.0 at [2,1]) first
        assert_eq!(dets.len(), 3);
        assert_eq!(objs.len(), 3);

        // First match should be the minimum value location [2,1]
        assert_eq!(dets[0], 2);
        assert_eq!(objs[0], 1);
    }

    #[test]
    fn test_minimum_matrix_value() {
        // Verify that all matches are below threshold
        let matrix = DMatrix::from_row_slice(3, 3, &[
            5.0, 3.0, 7.0,
            2.0, 9.0, 4.0,
            6.0, 1.0, 8.0,
        ]);

        // With threshold 2.5, only values ≤2.5 should match
        let (dets, objs) = match_detections_and_objects(&matrix, 2.5);

        // Should match [2,1]=1.0 and [1,0]=2.0
        assert_eq!(dets.len(), 2);
        assert_eq!(objs.len(), 2);

        // Greedy order: 1.0 first, then 2.0
        assert_eq!(dets[0], 2); // row 2
        assert_eq!(objs[0], 1); // col 1, value 1.0

        assert_eq!(dets[1], 1); // row 1
        assert_eq!(objs[1], 0); // col 0, value 2.0
    }
}
