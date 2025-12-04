//! Detection-to-object matching algorithms.

use nalgebra::DMatrix;

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

    // Greedy matching
    let mut matched_dets = Vec::new();
    let mut matched_objs = Vec::new();
    let mut used_dets = std::collections::HashSet::new();
    let mut used_objs = std::collections::HashSet::new();

    for (_dist, det_idx, obj_idx) in pairs {
        if used_dets.contains(&det_idx) || used_objs.contains(&obj_idx) {
            continue;
        }

        matched_dets.push(det_idx);
        matched_objs.push(obj_idx);
        used_dets.insert(det_idx);
        used_objs.insert(obj_idx);
    }

    (matched_dets, matched_objs)
}

/// Get unmatched indices from a match result.
pub fn get_unmatched(total: usize, matched: &[usize]) -> Vec<usize> {
    let matched_set: std::collections::HashSet<_> = matched.iter().cloned().collect();
    (0..total).filter(|i| !matched_set.contains(i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_empty() {
        let matrix = DMatrix::zeros(0, 0);
        let (dets, objs) = match_detections_and_objects(&matrix, 1.0);
        assert!(dets.is_empty());
        assert!(objs.is_empty());
    }

    #[test]
    fn test_match_single() {
        let matrix = DMatrix::from_row_slice(1, 1, &[0.5]);
        let (dets, objs) = match_detections_and_objects(&matrix, 1.0);
        assert_eq!(dets, vec![0]);
        assert_eq!(objs, vec![0]);
    }

    #[test]
    fn test_match_threshold() {
        let matrix = DMatrix::from_row_slice(1, 1, &[1.5]);
        let (dets, objs) = match_detections_and_objects(&matrix, 1.0);
        assert!(dets.is_empty());
        assert!(objs.is_empty());
    }

    #[test]
    fn test_match_greedy() {
        // Detection 0 is closer to Object 1, Detection 1 is closer to Object 0
        // But greedy should match based on smallest distance first
        let matrix = DMatrix::from_row_slice(2, 2, &[
            0.5, 0.1,  // det 0: closer to obj 1
            0.2, 0.6,  // det 1: closer to obj 0
        ]);
        let (dets, objs) = match_detections_and_objects(&matrix, 1.0);

        // Smallest is (0, 1) = 0.1, then (1, 0) = 0.2
        assert_eq!(dets.len(), 2);
        assert_eq!(objs.len(), 2);
    }

    #[test]
    fn test_get_unmatched() {
        let unmatched = get_unmatched(5, &[1, 3]);
        assert_eq!(unmatched, vec![0, 2, 4]);
    }
}
