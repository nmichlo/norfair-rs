//! NumPy-like array operations.

use nalgebra::DMatrix;
use crate::{Error, Result};

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
    fn test_flatten() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let flat = flatten(&matrix);
        assert_eq!(flat.len(), 4);
    }

    #[test]
    fn test_reshape() {
        let vec = nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let matrix = reshape(&vec, 2, 2).unwrap();
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);
    }
}
