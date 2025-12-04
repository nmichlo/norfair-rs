//! Kalman filter implementation ported from filterpy.
//!
//! This provides a standard Kalman filter with configurable state transition,
//! measurement, and noise matrices.

use nalgebra::{DMatrix, DVector};

/// Standard Kalman filter.
///
/// This is a port of filterpy's KalmanFilter class.
#[derive(Clone, Debug)]
pub struct KalmanFilter {
    /// State dimension
    pub dim_x: usize,
    /// Measurement dimension
    pub dim_z: usize,
    /// State vector
    pub x: DVector<f64>,
    /// State covariance matrix
    pub p: DMatrix<f64>,
    /// State transition matrix
    pub f: DMatrix<f64>,
    /// Measurement matrix
    pub h: DMatrix<f64>,
    /// Measurement noise covariance
    pub r: DMatrix<f64>,
    /// Process noise covariance
    pub q: DMatrix<f64>,
    /// Control input matrix (optional)
    pub b: Option<DMatrix<f64>>,
    // Working matrices (pre-allocated for efficiency)
    y: DVector<f64>,
    s: DMatrix<f64>,
    si: DMatrix<f64>,
    k: DMatrix<f64>,
}

impl KalmanFilter {
    /// Create a new Kalman filter.
    ///
    /// # Arguments
    /// * `dim_x` - State dimension
    /// * `dim_z` - Measurement dimension
    pub fn new(dim_x: usize, dim_z: usize) -> Self {
        Self {
            dim_x,
            dim_z,
            x: DVector::zeros(dim_x),
            p: DMatrix::identity(dim_x, dim_x),
            f: DMatrix::identity(dim_x, dim_x),
            h: DMatrix::zeros(dim_z, dim_x),
            r: DMatrix::identity(dim_z, dim_z),
            q: DMatrix::identity(dim_x, dim_x),
            b: None,
            y: DVector::zeros(dim_z),
            s: DMatrix::zeros(dim_z, dim_z),
            si: DMatrix::zeros(dim_z, dim_z),
            k: DMatrix::zeros(dim_x, dim_z),
        }
    }

    /// Predict the next state.
    ///
    /// # Arguments
    /// * `u` - Optional control input
    pub fn predict(&mut self, u: Option<&DVector<f64>>) {
        // x = F @ x + B @ u
        self.x = &self.f * &self.x;
        if let (Some(b), Some(u)) = (&self.b, u) {
            self.x += b * u;
        }

        // P = F @ P @ F.T + Q
        self.p = &self.f * &self.p * self.f.transpose() + &self.q;
    }

    /// Update the state with a measurement.
    ///
    /// # Arguments
    /// * `z` - Measurement vector
    /// * `r` - Optional measurement noise covariance (overrides self.r)
    /// * `h` - Optional measurement matrix (overrides self.h)
    pub fn update(
        &mut self,
        z: &DVector<f64>,
        r: Option<&DMatrix<f64>>,
        h: Option<&DMatrix<f64>>,
    ) {
        let r = r.unwrap_or(&self.r);
        let h = h.unwrap_or(&self.h);

        // y = z - H @ x (innovation)
        self.y = z - h * &self.x;

        // S = H @ P @ H.T + R (innovation covariance)
        self.s = h * &self.p * h.transpose() + r;

        // K = P @ H.T @ S^-1 (Kalman gain)
        self.si = self.s.clone().try_inverse().unwrap_or_else(|| {
            // Fallback: use pseudo-inverse or identity
            DMatrix::identity(self.dim_z, self.dim_z)
        });
        self.k = &self.p * h.transpose() * &self.si;

        // x = x + K @ y
        self.x = &self.x + &self.k * &self.y;

        // P = (I - K @ H) @ P
        let i = DMatrix::identity(self.dim_x, self.dim_x);
        self.p = (&i - &self.k * h) * &self.p;
    }

    /// Get the current state estimate.
    pub fn get_state(&self) -> &DVector<f64> {
        &self.x
    }

    /// Get the state covariance.
    pub fn get_covariance(&self) -> &DMatrix<f64> {
        &self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kalman_filter_create() {
        let kf = KalmanFilter::new(4, 2);
        assert_eq!(kf.dim_x, 4);
        assert_eq!(kf.dim_z, 2);
        assert_eq!(kf.x.len(), 4);
        assert_eq!(kf.p.nrows(), 4);
        assert_eq!(kf.p.ncols(), 4);
    }

    #[test]
    fn test_kalman_filter_predict() {
        let mut kf = KalmanFilter::new(4, 2);

        // Set up simple constant velocity model
        // State: [x, y, vx, vy]
        kf.f = DMatrix::from_row_slice(4, 4, &[
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);

        kf.x = DVector::from_vec(vec![1.0, 1.0, 0.5, 0.5]);

        kf.predict(None);

        // Position should have moved by velocity
        assert_relative_eq!(kf.x[0], 1.5, epsilon = 1e-10);
        assert_relative_eq!(kf.x[1], 1.5, epsilon = 1e-10);
        assert_relative_eq!(kf.x[2], 0.5, epsilon = 1e-10);
        assert_relative_eq!(kf.x[3], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_kalman_filter_update() {
        let mut kf = KalmanFilter::new(4, 2);

        // Measurement matrix: only observe position
        kf.h = DMatrix::from_row_slice(2, 4, &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ]);

        kf.x = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

        // Update with measurement at (1, 1)
        let z = DVector::from_vec(vec![1.0, 1.0]);
        kf.update(&z, None, None);

        // State should move towards measurement
        assert!(kf.x[0] > 0.0);
        assert!(kf.x[1] > 0.0);
    }
}
