//! Kalman filter implementations for object tracking.
//!
//! This module provides multiple filter implementations:
//! - `OptimizedKalmanFilter` - Simplified, fast covariance tracking
//! - `FilterPyKalmanFilter` - Full Kalman filter (filterpy-compatible)
//! - `NoFilter` - Baseline without prediction

mod traits;
mod optimized;
mod filterpy;
mod no_filter;

pub use traits::{Filter, FilterFactory};
pub use optimized::{OptimizedKalmanFilter, OptimizedKalmanFilterFactory};
pub use filterpy::{FilterPyKalmanFilter, FilterPyKalmanFilterFactory};
pub use no_filter::{NoFilter, NoFilterFactory};

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use approx::assert_relative_eq;

    // ===== Filter Comparison Tests =====

    #[test]
    fn test_filter_comparison_static_object() {
        // Both filters should produce similar results for a static object
        let initial_detection = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);

        let filterpy_factory = FilterPyKalmanFilterFactory::new(4.0, 0.1, 10.0);
        let optimized_factory = OptimizedKalmanFilterFactory::new(4.0, 0.1, 10.0, 0.0, 1.0);

        let mut filterpy = filterpy_factory.create_filter(&initial_detection);
        let mut optimized = optimized_factory.create_filter(&initial_detection);

        // Run same sequence on both
        for _ in 0..10 {
            filterpy.predict();
            optimized.predict();

            let measurement = DVector::from_vec(vec![1.0, 1.0]);
            filterpy.update(&measurement, None, None);
            optimized.update(&measurement, None, None);
        }

        let state_py = filterpy.get_state();
        let state_opt = optimized.get_state();

        // States should be very close (allowing for some numerical differences)
        assert_relative_eq!(state_opt[(0, 0)], state_py[(0, 0)], epsilon = 0.01);
        assert_relative_eq!(state_opt[(0, 1)], state_py[(0, 1)], epsilon = 0.01);
    }

    #[test]
    fn test_filter_comparison_moving_object() {
        // Both filters should produce similar results for a moving object
        let initial_detection = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);

        let filterpy_factory = FilterPyKalmanFilterFactory::new(4.0, 0.1, 10.0);
        let optimized_factory = OptimizedKalmanFilterFactory::new(4.0, 0.1, 10.0, 0.0, 1.0);

        let mut filterpy = filterpy_factory.create_filter(&initial_detection);
        let mut optimized = optimized_factory.create_filter(&initial_detection);

        // Simulate a moving object
        let positions = vec![
            (1.0, 1.0),
            (1.0, 2.0),
            (1.0, 3.0),
            (1.0, 4.0),
            (1.0, 5.0),
        ];

        for (x, y) in positions {
            let measurement = DVector::from_vec(vec![x, y]);
            filterpy.update(&measurement, None, None);
            optimized.update(&measurement, None, None);

            filterpy.predict();
            optimized.predict();
        }

        let state_py = filterpy.get_state();
        let state_opt = optimized.get_state();

        // States should be reasonably close (allowing for algorithmic differences)
        assert_relative_eq!(state_opt[(0, 0)], state_py[(0, 0)], epsilon = 0.2);
        assert_relative_eq!(state_opt[(0, 1)], state_py[(0, 1)], epsilon = 0.2);
    }

    // ===== Partial Measurement Tests (H matrix) =====

    #[test]
    fn test_filterpy_kalman_partial_measurement() {
        let initial_detection = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = FilterPyKalmanFilterFactory::new(4.0, 0.1, 10.0);
        let mut filter = factory.create_filter(&initial_detection);

        // Create H matrix that only observes the first dimension
        let h = DMatrix::from_row_slice(2, 4, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, // Second dimension not observed
        ]);

        // Second value (100.0) should be ignored since H[1,1]=0
        let measurement = DVector::from_vec(vec![2.0, 100.0]);
        filter.update(&measurement, None, Some(&h));

        let state = filter.get_state();
        // First position should be updated towards 2.0
        assert!(state[(0, 0)] > 1.5, "position x should move toward 2.0, got {}", state[(0, 0)]);
        // Second position should stay close to 1.0 (not affected by the 100.0)
        assert_relative_eq!(state[(0, 1)], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_optimized_kalman_partial_measurement() {
        let initial_detection = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = OptimizedKalmanFilterFactory::new(4.0, 0.1, 10.0, 0.0, 1.0);
        let mut filter = factory.create_filter(&initial_detection);

        // Create H matrix that only observes the first dimension
        let h = DMatrix::from_row_slice(2, 4, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, // Second dimension not observed
        ]);

        // Second value (100.0) should be ignored since H[1,1]=0
        let measurement = DVector::from_vec(vec![2.0, 100.0]);
        filter.update(&measurement, None, Some(&h));

        let state = filter.get_state();
        // First position should be updated towards 2.0
        assert!(state[(0, 0)] > 1.5, "position x should move toward 2.0, got {}", state[(0, 0)]);
        // Second position should stay close to 1.0 (not affected by the 100.0)
        assert_relative_eq!(state[(0, 1)], 1.0, epsilon = 0.1);
    }

    // ===== Multi-point Tests =====

    #[test]
    fn test_filters_multipoint() {
        // Test with 2 points, 2D each (e.g., bounding box corners)
        let initial_detection = DMatrix::from_row_slice(2, 2, &[
            0.0, 0.0,
            1.0, 1.0,
        ]);

        let filterpy_factory = FilterPyKalmanFilterFactory::new(4.0, 0.1, 10.0);
        let optimized_factory = OptimizedKalmanFilterFactory::new(4.0, 0.1, 10.0, 0.0, 1.0);

        let mut filterpy = filterpy_factory.create_filter(&initial_detection);
        let mut optimized = optimized_factory.create_filter(&initial_detection);

        // Should have dim_z = 4 for 2 points x 2 dims
        assert_eq!(filterpy.dim_z(), 4);
        assert_eq!(optimized.dim_z(), 4);

        // Update with new measurements (flattened: [p1_x, p1_y, p2_x, p2_y])
        let measurement = DVector::from_vec(vec![0.1, 0.1, 1.1, 1.1]);
        filterpy.update(&measurement, None, None);
        optimized.update(&measurement, None, None);

        // Both should handle multi-point correctly
        let state_py = filterpy.get_state();
        let state_opt = optimized.get_state();

        // Check that both filters updated properly
        assert_relative_eq!(state_py[(0, 0)], 0.1, epsilon = 0.1);
        assert_relative_eq!(state_py[(0, 1)], 0.1, epsilon = 0.1);
        assert_relative_eq!(state_py[(1, 0)], 1.1, epsilon = 0.1);
        assert_relative_eq!(state_py[(1, 1)], 1.1, epsilon = 0.1);

        assert_relative_eq!(state_opt[(0, 0)], 0.1, epsilon = 0.1);
        assert_relative_eq!(state_opt[(0, 1)], 0.1, epsilon = 0.1);
        assert_relative_eq!(state_opt[(1, 0)], 1.1, epsilon = 0.1);
        assert_relative_eq!(state_opt[(1, 1)], 1.1, epsilon = 0.1);
    }

    #[test]
    fn test_nofilter_multipoint() {
        // Test NoFilter with multiple points
        let initial_detection = DMatrix::from_row_slice(2, 2, &[
            0.0, 0.0,
            1.0, 1.0,
        ]);

        let factory = NoFilterFactory::new();
        let mut filter = factory.create_filter(&initial_detection);

        assert_eq!(filter.dim_z(), 4);

        let measurement = DVector::from_vec(vec![0.5, 0.5, 1.5, 1.5]);
        filter.update(&measurement, None, None);

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 0.5, epsilon = 1e-10);
        assert_relative_eq!(state[(0, 1)], 0.5, epsilon = 1e-10);
        assert_relative_eq!(state[(1, 0)], 1.5, epsilon = 1e-10);
        assert_relative_eq!(state[(1, 1)], 1.5, epsilon = 1e-10);
    }
}
