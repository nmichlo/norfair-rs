//! FilterPy-compatible Kalman filter implementation.
//!
//! This filter maintains full covariance matrices, matching the behavior
//! of Python's filterpy library.

use super::traits::{Filter, FilterFactory};
use crate::internal::filterpy::KalmanFilter as InternalKalmanFilter;
use nalgebra::{DMatrix, DVector};

/// FilterPy-compatible Kalman filter.
///
/// This maintains full covariance matrices and provides behavior
/// equivalent to filterpy.kalman.KalmanFilter.
#[derive(Clone, Debug)]
pub struct FilterPyKalmanFilter {
    kf: InternalKalmanFilter,
    n_points: usize,
    n_dims: usize,
}

impl FilterPyKalmanFilter {
    /// Create a new FilterPy-compatible Kalman filter.
    pub fn new(initial_detection: &DMatrix<f64>, r: f64, q: f64, p: f64) -> Self {
        let n_points = initial_detection.nrows();
        let n_dims = initial_detection.ncols();
        let dim_z = n_points * n_dims;
        let dim_x = dim_z * 2; // position + velocity

        let mut kf = InternalKalmanFilter::new(dim_x, dim_z);

        // Initialize state: [positions..., velocities...]
        for i in 0..n_points {
            for j in 0..n_dims {
                kf.x[i * n_dims + j] = initial_detection[(i, j)];
            }
        }
        // Velocities start at 0

        // State transition matrix F: constant velocity model
        // [I, I]
        // [0, I]
        kf.f = DMatrix::identity(dim_x, dim_x);
        for i in 0..dim_z {
            kf.f[(i, dim_z + i)] = 1.0;
        }

        // Measurement matrix H: observe only position
        // [I, 0]
        kf.h = DMatrix::zeros(dim_z, dim_x);
        for i in 0..dim_z {
            kf.h[(i, i)] = 1.0;
        }

        // Measurement noise R
        kf.r = DMatrix::identity(dim_z, dim_z) * r;

        // Process noise Q
        kf.q = DMatrix::identity(dim_x, dim_x) * q;

        // Initial covariance P
        kf.p = DMatrix::zeros(dim_x, dim_x);
        // Position variance
        for i in 0..dim_z {
            kf.p[(i, i)] = p;
        }
        // Velocity variance (smaller)
        for i in dim_z..dim_x {
            kf.p[(i, i)] = 1.0;
        }

        Self {
            kf,
            n_points,
            n_dims,
        }
    }
}

impl Filter for FilterPyKalmanFilter {
    fn predict(&mut self) {
        self.kf.predict(None);
    }

    fn update(
        &mut self,
        measurement: &DVector<f64>,
        r: Option<&DMatrix<f64>>,
        h: Option<&DMatrix<f64>>,
    ) {
        self.kf.update(measurement, r, h);
    }

    fn get_state(&self) -> DMatrix<f64> {
        // Return positions reshaped to (n_points, n_dims)
        let mut result = DMatrix::zeros(self.n_points, self.n_dims);
        for i in 0..self.n_points {
            for j in 0..self.n_dims {
                result[(i, j)] = self.kf.x[i * self.n_dims + j];
            }
        }
        result
    }

    fn get_state_vector(&self) -> &DVector<f64> {
        &self.kf.x
    }

    fn set_state_vector(&mut self, x: &DVector<f64>) {
        self.kf.x.copy_from(x);
    }

    fn dim_z(&self) -> usize {
        self.kf.dim_z
    }

    fn dim_x(&self) -> usize {
        self.kf.dim_x
    }
}

/// Factory for creating FilterPyKalmanFilter instances.
#[derive(Clone, Debug)]
pub struct FilterPyKalmanFilterFactory {
    r: f64,
    q: f64,
    p: f64,
}

impl FilterPyKalmanFilterFactory {
    /// Create a new factory with the specified parameters.
    ///
    /// # Arguments
    /// * `r` - Measurement noise variance
    /// * `q` - Process noise variance
    /// * `p` - Initial position variance
    pub fn new(r: f64, q: f64, p: f64) -> Self {
        Self { r, q, p }
    }

    /// Get measurement noise variance.
    #[inline(always)]
    pub fn r(&self) -> f64 {
        self.r
    }

    /// Get process noise variance.
    #[inline(always)]
    pub fn q(&self) -> f64 {
        self.q
    }

    /// Get initial position variance.
    #[inline(always)]
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Default for FilterPyKalmanFilterFactory {
    fn default() -> Self {
        Self::new(4.0, 0.1, 10.0)
    }
}

impl FilterFactory for FilterPyKalmanFilterFactory {
    fn create_filter(&self, initial_detection: &DMatrix<f64>) -> Box<dyn Filter> {
        Box::new(FilterPyKalmanFilter::new(
            initial_detection,
            self.r,
            self.q,
            self.p,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_filterpy_kalman_create() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = FilterPyKalmanFilterFactory::default();
        let filter = factory.create_filter(&initial);

        assert_eq!(filter.dim_z(), 2);
        assert_eq!(filter.dim_x(), 4);

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(state[(0, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_filterpy_kalman_static_object() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = FilterPyKalmanFilterFactory::default();
        let mut filter = factory.create_filter(&initial);

        for _ in 0..5 {
            filter.predict();
            let measurement = DVector::from_vec(vec![1.0, 1.0]);
            filter.update(&measurement, None, None);
        }

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 1.0, epsilon = 0.1);
        assert_relative_eq!(state[(0, 1)], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_filterpy_kalman_moving_object() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = FilterPyKalmanFilterFactory::default();
        let mut filter = factory.create_filter(&initial);

        let positions = vec![1.0, 2.0, 3.0, 4.0];
        for y in positions {
            let measurement = DVector::from_vec(vec![1.0, y]);
            filter.update(&measurement, None, None);
            filter.predict();
        }

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 1.0, epsilon = 0.5);
        assert!(state[(0, 1)] > 3.0 && state[(0, 1)] <= 5.0);
    }
}
