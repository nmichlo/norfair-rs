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

    #[test]
    fn test_cdist_euclidean() {
        let xa = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 0.0]);
        let xb = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 1.0]);

        let result = cdist(&xa, &xb, "euclidean");

        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        assert_relative_eq!(result[(0, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[(0, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[(1, 1)], 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_cdist_manhattan() {
        let xa = DMatrix::from_row_slice(1, 2, &[0.0, 0.0]);
        let xb = DMatrix::from_row_slice(1, 2, &[3.0, 4.0]);

        let result = cdist(&xa, &xb, "manhattan");

        assert_relative_eq!(result[(0, 0)], 7.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cdist_cosine() {
        let xa = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let xb = DMatrix::from_row_slice(1, 2, &[0.0, 1.0]);

        let result = cdist(&xa, &xb, "cosine");

        // Orthogonal vectors have cosine distance of 1
        assert_relative_eq!(result[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cdist_chebyshev() {
        let xa = DMatrix::from_row_slice(1, 2, &[0.0, 0.0]);
        let xb = DMatrix::from_row_slice(1, 2, &[3.0, 4.0]);

        let result = cdist(&xa, &xb, "chebyshev");

        assert_relative_eq!(result[(0, 0)], 4.0, epsilon = 1e-10);
    }
}
