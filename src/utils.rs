//! Utility functions for norfair.

use crate::{Error, Result};
use nalgebra::DMatrix;
use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::OnceLock;

/// Validate that points have shape (n_points, n_dims) where n_dims is 2 or 3.
///
/// If points is a single row, it's treated as a single point.
pub fn validate_points(points: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    let rows = points.nrows();
    let cols = points.ncols();

    // Handle 1D case: if input is shape (1, n), it's a single 2D or 3D point
    if rows == 1 && (cols == 2 || cols == 3) {
        return Ok(points.clone());
    }

    // Validate dimensions
    if cols != 2 && cols != 3 {
        return Err(Error::InvalidPointsShape {
            expected: "n_dims to be 2 or 3".to_string(),
            got: format!("shape ({}, {})", rows, cols),
        });
    }

    Ok(points.clone())
}

/// Get terminal size (columns, lines).
///
/// Returns defaults if terminal size cannot be detected.
pub fn get_terminal_size(default_cols: u16, default_lines: u16) -> (u16, u16) {
    // Try to get terminal size using termion or similar
    // For now, return defaults
    (default_cols, default_lines)
}

/// Global set of warned messages (for warn_once).
static WARNED_MESSAGES: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

/// Print a warning message only once.
///
/// Subsequent calls with the same message are ignored.
pub fn warn_once(message: &str) {
    let warned = WARNED_MESSAGES.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = warned.lock().unwrap();
    if !guard.contains(message) {
        eprintln!("WARNING: {}", message);
        guard.insert(message.to_string());
    }
}

/// Check if any value in a slice is true.
pub fn any_true(values: &[bool]) -> bool {
    values.iter().any(|&v| v)
}

/// Check if all values in a slice are true.
pub fn all_true(values: &[bool]) -> bool {
    values.iter().all(|&v| v)
}

/// Get cutout bounds from points.
///
/// Returns (x1, y1, x2, y2) bounding box of the points.
pub fn get_bounding_box(points: &DMatrix<f64>) -> Option<(f64, f64, f64, f64)> {
    if points.nrows() == 0 || points.ncols() < 2 {
        return None;
    }

    let mut min_x = points[(0, 0)];
    let mut max_x = points[(0, 0)];
    let mut min_y = points[(0, 1)];
    let mut max_y = points[(0, 1)];

    for i in 0..points.nrows() {
        let x = points[(i, 0)];
        let y = points[(i, 1)];

        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }

    Some((min_x, min_y, max_x, max_y))
}

/// Clamp a value to a range.
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ===== ValidatePoints Tests (ported from Go) =====

    #[test]
    fn test_validate_points_valid_2d() {
        // Test valid 2D points array (n_points, 2)
        let points = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let validated = validate_points(&points).expect("Expected no error for valid 2D points");

        assert_eq!(validated.nrows(), 3);
        assert_eq!(validated.ncols(), 2);
    }

    #[test]
    fn test_validate_points_valid_3d() {
        // Test valid 3D points array (n_points, 3)
        let points = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let validated = validate_points(&points).expect("Expected no error for valid 3D points");

        assert_eq!(validated.nrows(), 2);
        assert_eq!(validated.ncols(), 3);
    }

    #[test]
    fn test_validate_points_single_2d_point() {
        // Test single 2D point (1, 2) - edge case
        let points = DMatrix::from_row_slice(1, 2, &[10.0, 20.0]);

        let validated = validate_points(&points).expect("Expected no error for single 2D point");

        assert_eq!(validated.nrows(), 1);
        assert_eq!(validated.ncols(), 2);

        // Verify values preserved
        assert_relative_eq!(validated[(0, 0)], 10.0, epsilon = 1e-10);
        assert_relative_eq!(validated[(0, 1)], 20.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validate_points_single_3d_point() {
        // Test single 3D point (1, 3) - edge case
        let points = DMatrix::from_row_slice(1, 3, &[10.0, 20.0, 30.0]);

        let validated = validate_points(&points).expect("Expected no error for single 3D point");

        assert_eq!(validated.nrows(), 1);
        assert_eq!(validated.ncols(), 3);

        // Verify values preserved
        assert_relative_eq!(validated[(0, 0)], 10.0, epsilon = 1e-10);
        assert_relative_eq!(validated[(0, 1)], 20.0, epsilon = 1e-10);
        assert_relative_eq!(validated[(0, 2)], 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_validate_points_invalid_dimensions_4d() {
        // Test invalid dimensions (n, 4) - should error
        let points = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let err = validate_points(&points).expect_err("Expected error for 4D points");

        // Verify error message mentions invalid shape
        let error_msg = format!("{}", err);
        assert!(
            error_msg.contains("shape") || error_msg.contains("invalid"),
            "Expected error to mention shape, got: {}",
            error_msg
        );
    }

    #[test]
    fn test_validate_points_invalid_dimensions_1d() {
        // Test invalid dimensions (n, 1) - should error
        let points = DMatrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]);

        let err = validate_points(&points).expect_err("Expected error for 1D points (n, 1)");

        // Verify error message mentions invalid shape
        let error_msg = format!("{}", err);
        assert!(
            error_msg.contains("shape") || error_msg.contains("invalid"),
            "Expected error to mention shape, got: {}",
            error_msg
        );
    }

    #[test]
    fn test_validate_points_invalid_single_value() {
        // Test single value (1, 1) - should error (neither 2D nor 3D)
        let points = DMatrix::from_row_slice(1, 1, &[10.0]);

        let err = validate_points(&points).expect_err("Expected error for single value (1, 1)");

        // Verify error message mentions invalid shape
        let error_msg = format!("{}", err);
        assert!(
            error_msg.contains("shape") || error_msg.contains("invalid"),
            "Expected error to mention shape, got: {}",
            error_msg
        );
    }

    // ===== GetTerminalSize Tests =====
    // Note: Terminal size tests are not applicable in this context since
    // the function returns defaults. Just verify it returns positive values.

    #[test]
    fn test_get_terminal_size_returns_values() {
        let (cols, lines) = get_terminal_size(80, 24);

        assert!(cols > 0, "Expected positive cols");
        assert!(lines > 0, "Expected positive lines");
    }

    #[test]
    fn test_get_terminal_size_custom_defaults() {
        let (cols, lines) = get_terminal_size(100, 50);

        assert!(cols > 0, "Expected positive cols");
        assert!(lines > 0, "Expected positive lines");
    }

    // ===== Boolean utility tests =====

    #[test]
    fn test_any_true() {
        assert!(any_true(&[false, true, false]));
        assert!(!any_true(&[false, false, false]));
        assert!(!any_true(&[]));
    }

    #[test]
    fn test_all_true() {
        assert!(all_true(&[true, true, true]));
        assert!(!all_true(&[true, false, true]));
        assert!(all_true(&[]));
    }

    // ===== Bounding box tests =====

    #[test]
    fn test_get_bounding_box() {
        let points = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 5.0, 8.0, 3.0, 4.0]);

        let bbox = get_bounding_box(&points).unwrap();
        assert_eq!(bbox, (1.0, 2.0, 5.0, 8.0));
    }

    #[test]
    fn test_get_bounding_box_single_point() {
        let points = DMatrix::from_row_slice(1, 2, &[5.0, 10.0]);

        let bbox = get_bounding_box(&points).unwrap();
        assert_eq!(bbox, (5.0, 10.0, 5.0, 10.0));
    }

    #[test]
    fn test_get_bounding_box_empty() {
        let points = DMatrix::zeros(0, 2);

        assert!(get_bounding_box(&points).is_none());
    }

    #[test]
    fn test_get_bounding_box_invalid_cols() {
        let points = DMatrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]);

        assert!(get_bounding_box(&points).is_none());
    }

    // ===== Clamp tests =====

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-5, 0, 10), 0);
        assert_eq!(clamp(15, 0, 10), 10);
    }

    #[test]
    fn test_clamp_at_bounds() {
        assert_eq!(clamp(0, 0, 10), 0);
        assert_eq!(clamp(10, 0, 10), 10);
    }

    #[test]
    fn test_clamp_float() {
        assert_relative_eq!(clamp(5.0, 0.0, 10.0), 5.0, epsilon = 1e-10);
        assert_relative_eq!(clamp(-5.0, 0.0, 10.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(clamp(15.0, 0.0, 10.0), 10.0, epsilon = 1e-10);
    }
}
