//! NumPy-like array operations.
//!
//! Ported from numpy:
//! - linspace: numpy.linspace
//! - validate_points, flatten, reshape: array utilities

use nalgebra::DMatrix;
use crate::{Error, Result};

/// Generate `n` evenly spaced values between `start` and `end` (inclusive).
///
/// This is a port of numpy.linspace which returns evenly spaced numbers over
/// a specified interval.
///
/// # Arguments
/// * `start` - Starting value of the sequence
/// * `end` - End value of the sequence
/// * `n` - Number of samples to generate
///
/// # Returns
/// Vector of `n` evenly spaced float64 values
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }

    let mut result = Vec::with_capacity(n);
    let step = (end - start) / (n - 1) as f64;

    for i in 0..n {
        result.push(start + i as f64 * step);
    }

    // Ensure endpoint is exact (avoid floating point drift)
    result[n - 1] = end;

    result
}

/// Validate and normalize points to 2D shape (n_points, n_dims).
///
/// Accepts:
/// - 1D array [x, y] -> [[x, y]] (1 point)
/// - 2D array [[x1, y1], [x2, y2], ...] (n points)
pub fn validate_points(points: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let (rows, cols) = points.shape();

    if rows == 0 || cols == 0 {
        return Err(Error::InvalidPointsShape {
            expected: "(n_points, n_dims)".to_string(),
            got: format!("({}, {})", rows, cols),
        });
    }

    // If it's a row vector (1 x n), treat as single point
    if rows == 1 && cols >= 2 {
        return Ok(points.clone());
    }

    // If it's (n x d), it's already correct format
    if cols >= 2 {
        return Ok(points.clone());
    }

    Err(Error::InvalidPointsShape {
        expected: "(n_points, n_dims) where n_dims >= 2".to_string(),
        got: format!("({}, {})", rows, cols),
    })
}

/// Flatten a matrix to a column vector (column-major order).
pub fn flatten(matrix: &DMatrix<f64>) -> nalgebra::DVector<f64> {
    let data: Vec<f64> = matrix.iter().cloned().collect();
    nalgebra::DVector::from_vec(data)
}

/// Reshape a vector into a matrix.
pub fn reshape(vec: &nalgebra::DVector<f64>, rows: usize, cols: usize) -> Result<DMatrix<f64>> {
    if vec.len() != rows * cols {
        return Err(Error::InvalidPointsShape {
            expected: format!("vector of length {}", rows * cols),
            got: format!("vector of length {}", vec.len()),
        });
    }

    Ok(DMatrix::from_vec(rows, cols, vec.iter().cloned().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== linspace tests =====

    #[test]
    fn test_linspace_basic() {
        // Test case: 5 values from 0 to 10
        let result = linspace(0.0, 10.0, 5);
        let expected = [0.0, 2.5, 5.0, 7.5, 10.0];
        assert_eq!(result.len(), expected.len());
        for (i, &val) in result.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-10, "Linspace value at {}", i);
        }
    }

    #[test]
    fn test_linspace_two_points() {
        let result = linspace(1.0, 10.0, 2);
        let expected = [1.0, 10.0];
        assert_eq!(result.len(), expected.len());
        assert!((result[0] - expected[0]).abs() < 1e-10, "Start value");
        assert!((result[1] - expected[1]).abs() < 1e-10, "End value");
    }

    #[test]
    fn test_linspace_single_point() {
        let result = linspace(5.0, 10.0, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linspace_zero() {
        let result = linspace(0.0, 10.0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_linspace_negative() {
        let result = linspace(-10.0, 10.0, 5);
        let expected = [-10.0, -5.0, 0.0, 5.0, 10.0];
        assert_eq!(result.len(), expected.len());
        for (i, &val) in result.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-10, "Negative range value at {}", i);
        }
    }

    #[test]
    fn test_linspace_reverse_range() {
        let result = linspace(10.0, 0.0, 5);
        let expected = [10.0, 7.5, 5.0, 2.5, 0.0];
        assert_eq!(result.len(), expected.len());
        for (i, &val) in result.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-10, "Reverse range value at {}", i);
        }
    }

    #[test]
    fn test_linspace_floating_point() {
        let result = linspace(0.1, 0.9, 5);
        let expected = [0.1, 0.3, 0.5, 0.7, 0.9];
        assert_eq!(result.len(), expected.len());
        for (i, &val) in result.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-10, "Floating-point value at {}", i);
        }
    }

    #[test]
    fn test_linspace_large_n() {
        let result = linspace(0.0, 100.0, 101);
        assert_eq!(result.len(), 101);

        // Verify first, middle, and last values
        assert!((result[0] - 0.0).abs() < 1e-10, "Start value");
        assert!((result[50] - 50.0).abs() < 1e-10, "Middle value");
        assert!((result[100] - 100.0).abs() < 1e-10, "End value");

        // Verify all values are monotonically increasing
        for i in 1..result.len() {
            assert!(
                result[i] > result[i - 1],
                "Values not monotonically increasing: result[{}]={}, result[{}]={}",
                i - 1, result[i - 1], i, result[i]
            );
        }
    }

    #[test]
    fn test_linspace_endpoint_exact() {
        // Test with values that might accumulate floating-point error
        let result = linspace(0.0, 1.0, 100);
        assert_eq!(result.len(), 100);

        // Endpoint should be exactly 1.0, not 0.9999999...
        assert_eq!(result[99], 1.0, "Endpoint should be exactly 1.0, got {:.20}", result[99]);
    }

    #[test]
    fn test_linspace_zero_range() {
        let result = linspace(5.0, 5.0, 10);
        assert_eq!(result.len(), 10);

        // All values should be exactly 5
        for (i, &val) in result.iter().enumerate() {
            assert!((val - 5.0).abs() < 1e-10, "Zero range value at {}", i);
        }
    }

    #[test]
    fn test_linspace_small_interval() {
        let result = linspace(0.0, 1e-6, 5);
        assert_eq!(result.len(), 5);

        // Verify start and end
        assert!((result[0] - 0.0).abs() < 1e-15, "Start value");
        assert!((result[4] - 1e-6).abs() < 1e-15, "End value");

        // Verify spacing
        let expected_step = 0.25e-6;
        for i in 1..(result.len() - 1) {
            let expected = i as f64 * expected_step;
            assert!((result[i] - expected).abs() < 1e-15, "Small interval value at {}", i);
        }
    }

    #[test]
    fn test_linspace_large_interval() {
        let result = linspace(0.0, 1e10, 5);
        assert_eq!(result.len(), 5);

        let expected = [0.0, 2.5e9, 5e9, 7.5e9, 1e10];
        for (i, &val) in result.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-3, "Large interval value at {}", i);
        }
    }

    #[test]
    fn test_linspace_matches_numpy_behavior() {
        // Test cases that match numpy.linspace behavior
        let test_cases: Vec<(f64, f64, usize, Vec<f64>)> = vec![
            (0.0, 10.0, 5, vec![0.0, 2.5, 5.0, 7.5, 10.0]),
            (-5.0, 5.0, 11, vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            (0.0, 1.0, 3, vec![0.0, 0.5, 1.0]),
            (10.0, 0.0, 6, vec![10.0, 8.0, 6.0, 4.0, 2.0, 0.0]),
            (0.0, 0.0, 5, vec![0.0, 0.0, 0.0, 0.0, 0.0]),
        ];

        for (start, end, n, expected) in test_cases {
            let result = linspace(start, end, n);

            assert_eq!(
                result.len(),
                expected.len(),
                "Linspace({}, {}, {}): expected length {}, got {}",
                start, end, n, expected.len(), result.len()
            );

            for (i, &val) in result.iter().enumerate() {
                assert!(
                    (val - expected[i]).abs() < 1e-10,
                    "Linspace({}, {}, {})[{}]: expected {}, got {}",
                    start, end, n, i, expected[i], val
                );
            }
        }
    }

    #[test]
    fn test_linspace_consistency() {
        let result = linspace(0.0, 100.0, 11);
        assert_eq!(result.len(), 11);

        // Calculate differences between consecutive values
        for i in 1..(result.len() - 1) {
            let diff = result[i] - result[i - 1];
            assert!((diff - 10.0).abs() < 1e-10, "Consistent spacing at {}", i);
        }
    }

    // ===== validate_points tests =====

    #[test]
    fn test_validate_points_2d() {
        let points = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let result = validate_points(&points).unwrap();
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_validate_points_1d() {
        let points = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        let result = validate_points(&points).unwrap();
        assert_eq!(result.nrows(), 1);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_validate_points_3d() {
        let points = DMatrix::from_row_slice(3, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = validate_points(&points).unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 3);
    }

    #[test]
    fn test_validate_points_invalid_single_value() {
        let points = DMatrix::from_row_slice(1, 1, &[1.0]);
        let result = validate_points(&points);
        assert!(result.is_err());
    }

    // ===== flatten tests =====

    #[test]
    fn test_flatten() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let flat = flatten(&matrix);
        assert_eq!(flat.len(), 4);
    }

    // ===== reshape tests =====

    #[test]
    fn test_reshape() {
        let vec = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let matrix = reshape(&vec, 2, 2).unwrap();
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);
    }

    #[test]
    fn test_reshape_invalid_size() {
        let vec = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = reshape(&vec, 2, 2);
        assert!(result.is_err());
    }
}
