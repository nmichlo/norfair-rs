//! Utility functions for norfair.

use std::collections::HashSet;
use std::sync::OnceLock;
use std::sync::Mutex;
use nalgebra::DMatrix;
use crate::{Error, Result};

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

        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
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

    #[test]
    fn test_validate_points_valid() {
        let points = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(validate_points(&points).is_ok());

        let points_3d = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(validate_points(&points_3d).is_ok());
    }

    #[test]
    fn test_validate_points_invalid() {
        let points = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(validate_points(&points).is_err());
    }

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

    #[test]
    fn test_get_bounding_box() {
        let points = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            5.0, 8.0,
            3.0, 4.0,
        ]);

        let bbox = get_bounding_box(&points).unwrap();
        assert_eq!(bbox, (1.0, 2.0, 5.0, 8.0));
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-5, 0, 10), 0);
        assert_eq!(clamp(15, 0, 10), 10);
    }
}
