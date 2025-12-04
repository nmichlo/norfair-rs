//! Distance computation functions ported from scipy.spatial.distance
//!
//! This module provides cdist-like functionality for computing pairwise distances.

use nalgebra::DMatrix;

/// Compute pairwise distances between two sets of vectors using the specified metric.
///
/// # Arguments
/// * `xa` - First set of vectors (n_samples_a x n_features)
/// * `xb` - Second set of vectors (n_samples_b x n_features)
/// * `metric` - Distance metric name ("euclidean", "manhattan", "cosine", etc.)
///
/// # Returns
/// Distance matrix of shape (n_samples_a x n_samples_b)
pub fn cdist(xa: &DMatrix<f64>, xb: &DMatrix<f64>, metric: &str) -> DMatrix<f64> {
    let n_a = xa.nrows();
    let n_b = xb.nrows();
    let n_features = xa.ncols();

    assert_eq!(xa.ncols(), xb.ncols(), "Feature dimensions must match");

    let mut result = DMatrix::zeros(n_a, n_b);

    for i in 0..n_a {
        for j in 0..n_b {
            let dist = match metric {
                "euclidean" => {
                    let mut sum = 0.0;
                    for k in 0..n_features {
                        let diff = xa[(i, k)] - xb[(j, k)];
                        sum += diff * diff;
                    }
                    sum.sqrt()
                }
                "sqeuclidean" | "squared_euclidean" => {
                    let mut sum = 0.0;
                    for k in 0..n_features {
                        let diff = xa[(i, k)] - xb[(j, k)];
                        sum += diff * diff;
                    }
                    sum
                }
                "manhattan" | "cityblock" => {
                    let mut sum = 0.0;
                    for k in 0..n_features {
                        sum += (xa[(i, k)] - xb[(j, k)]).abs();
                    }
                    sum
                }
                "cosine" => {
                    let mut dot = 0.0;
                    let mut norm_a = 0.0;
                    let mut norm_b = 0.0;
                    for k in 0..n_features {
                        let a_k = xa[(i, k)];
                        let b_k = xb[(j, k)];
                        dot += a_k * b_k;
                        norm_a += a_k * a_k;
                        norm_b += b_k * b_k;
                    }
                    let norm_a = norm_a.sqrt();
                    let norm_b = norm_b.sqrt();
                    if norm_a == 0.0 || norm_b == 0.0 {
                        1.0
                    } else {
                        1.0 - dot / (norm_a * norm_b)
                    }
                }
                "chebyshev" => {
                    let mut max_diff = 0.0;
                    for k in 0..n_features {
                        let diff = (xa[(i, k)] - xb[(j, k)]).abs();
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                    max_diff
                }
                _ => panic!("Unknown metric: {}", metric),
            };
            result[(i, j)] = dist;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ===== Euclidean distance tests =====

    #[test]
    fn test_cdist_euclidean() {
        let xa = DMatrix::from_row_slice(2, 2, &[
            0.0, 0.0,
            1.0, 0.0,
        ]);
        let xb = DMatrix::from_row_slice(2, 2, &[
            0.0, 1.0,
            1.0, 1.0,
        ]);

        let result = cdist(&xa, &xb, "euclidean");

        // [0,0] to [0,1]: sqrt(1) = 1
        // [0,0] to [1,1]: sqrt(2)
        // [1,0] to [0,1]: sqrt(2)
        // [1,0] to [1,1]: sqrt(1) = 1
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        assert_relative_eq!(result[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[(0, 1)], 2.0_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(result[(1, 0)], 2.0_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(result[(1, 1)], 1.0, epsilon = 1e-10);
    }

    // ===== Manhattan distance tests =====

    #[test]
    fn test_cdist_manhattan() {
        let xa = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        let xb = DMatrix::from_row_slice(2, 3, &[
            1.0, 1.0, 1.0,
            2.0, 2.0, 2.0,
        ]);

        let result = cdist(&xa, &xb, "manhattan");

        // |1-1| + |2-1| + |3-1| = 0 + 1 + 2 = 3
        // |1-2| + |2-2| + |3-2| = 1 + 0 + 1 = 2
        // |4-1| + |5-1| + |6-1| = 3 + 4 + 5 = 12
        // |4-2| + |5-2| + |6-2| = 2 + 3 + 4 = 9
        assert_relative_eq!(result[(0, 0)], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[(0, 1)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 0)], 12.0, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 1)], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cdist_cityblock() {
        // Test alias
        let xa = DMatrix::from_row_slice(1, 2, &[0.0, 0.0]);
        let xb = DMatrix::from_row_slice(1, 2, &[3.0, 4.0]);

        let result = cdist(&xa, &xb, "cityblock");
        assert_relative_eq!(result[(0, 0)], 7.0, epsilon = 1e-10);
    }

    // ===== Cosine distance tests =====

    #[test]
    fn test_cdist_cosine_orthogonal() {
        // Orthogonal vectors: [1, 0] and [0, 1]
        let xa = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let xb = DMatrix::from_row_slice(1, 2, &[0.0, 1.0]);

        let result = cdist(&xa, &xb, "cosine");

        // Cosine similarity = 0, so cosine distance = 1
        assert_relative_eq!(result[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cdist_cosine_parallel() {
        // Parallel vectors: [1, 1] and [2, 2]
        let xa = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let xb = DMatrix::from_row_slice(1, 2, &[2.0, 2.0]);

        let result = cdist(&xa, &xb, "cosine");

        // Cosine similarity = 1, so cosine distance = 0
        assert_relative_eq!(result[(0, 0)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cdist_cosine_antiparallel() {
        // Anti-parallel vectors: [1, 0] and [-1, 0]
        let xa = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let xb = DMatrix::from_row_slice(1, 2, &[-1.0, 0.0]);

        let result = cdist(&xa, &xb, "cosine");

        // Cosine similarity = -1, so cosine distance = 1 - (-1) = 2
        assert_relative_eq!(result[(0, 0)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cdist_cosine_zero_vector() {
        // Zero vector case
        let xa = DMatrix::from_row_slice(1, 2, &[0.0, 0.0]);
        let xb = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);

        let result = cdist(&xa, &xb, "cosine");

        // Distance should be 1 for zero vector (undefined similarity treated as 1)
        assert_relative_eq!(result[(0, 0)], 1.0, epsilon = 1e-10);
    }

    // ===== Squared Euclidean distance tests =====

    #[test]
    fn test_cdist_sqeuclidean() {
        let xa = DMatrix::from_row_slice(2, 2, &[
            0.0, 0.0,
            3.0, 4.0,
        ]);
        let xb = DMatrix::from_row_slice(1, 2, &[0.0, 0.0]);

        let result = cdist(&xa, &xb, "sqeuclidean");

        // (0-0)^2 + (0-0)^2 = 0
        // (3-0)^2 + (4-0)^2 = 9 + 16 = 25
        assert_relative_eq!(result[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 0)], 25.0, epsilon = 1e-10);
    }

    // ===== Chebyshev distance tests =====

    #[test]
    fn test_cdist_chebyshev() {
        let xa = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            0.0, 0.0, 0.0,
        ]);
        let xb = DMatrix::from_row_slice(2, 3, &[
            2.0, 1.0, 1.0,
            5.0, 5.0, 5.0,
        ]);

        let result = cdist(&xa, &xb, "chebyshev");

        // max(|1-2|, |2-1|, |3-1|) = max(1, 1, 2) = 2
        // max(|1-5|, |2-5|, |3-5|) = max(4, 3, 2) = 4
        // max(|0-2|, |0-1|, |0-1|) = max(2, 1, 1) = 2
        // max(|0-5|, |0-5|, |0-5|) = max(5, 5, 5) = 5
        assert_relative_eq!(result[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[(0, 1)], 4.0, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 1)], 5.0, epsilon = 1e-10);
    }

    // ===== Different dimensions tests =====

    #[test]
    fn test_cdist_different_row_counts() {
        // Different number of rows is OK
        let xa = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let xb = DMatrix::from_row_slice(2, 2, &[
            0.0, 0.0,
            1.0, 1.0,
        ]);

        let result = cdist(&xa, &xb, "euclidean");

        // Should produce 3x2 matrix
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    #[should_panic(expected = "Feature dimensions must match")]
    fn test_cdist_panic_on_mismatched_columns() {
        let xa = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let xb = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        cdist(&xa, &xb, "euclidean");
    }

    #[test]
    #[should_panic(expected = "Unknown metric")]
    fn test_cdist_panic_on_unsupported_metric() {
        let xa = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let xb = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);

        cdist(&xa, &xb, "unsupported_metric");
    }

    // ===== Single vector tests =====

    #[test]
    fn test_cdist_single_vectors() {
        let xa = DMatrix::from_row_slice(1, 3, &[1.0, 2.0, 3.0]);
        let xb = DMatrix::from_row_slice(1, 3, &[4.0, 5.0, 6.0]);

        let result = cdist(&xa, &xb, "euclidean");

        // sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(9 + 9 + 9) = sqrt(27)
        let expected = 27.0_f64.sqrt();
        assert_relative_eq!(result[(0, 0)], expected, epsilon = 1e-10);
    }

    // ===== Identical vectors tests =====

    #[test]
    fn test_cdist_identical_vectors() {
        let xa = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        let xb = DMatrix::from_row_slice(2, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);

        for metric in &["euclidean", "manhattan", "sqeuclidean", "chebyshev"] {
            let result = cdist(&xa, &xb, metric);

            // Diagonal should be all zeros
            for i in 0..2 {
                assert!(
                    result[(i, i)].abs() < 1e-10,
                    "Distance between identical vectors should be 0 for {}, got {} at ({},{})",
                    metric, result[(i, i)], i, i
                );
            }
        }
    }
}
