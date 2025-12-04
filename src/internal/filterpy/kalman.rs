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
        // Initialize H matrix with identity in measurement dimensions
        let mut h = DMatrix::zeros(dim_z, dim_x);
        for i in 0..dim_z.min(dim_x) {
            h[(i, i)] = 1.0;
        }

        Self {
            dim_x,
            dim_z,
            x: DVector::zeros(dim_x),
            p: DMatrix::identity(dim_x, dim_x),
            f: DMatrix::identity(dim_x, dim_x),
            h,
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

    // ===== Initialization tests =====

    #[test]
    fn test_kalman_filter_create() {
        let kf = KalmanFilter::new(4, 2);

        // Verify dimensions
        assert_eq!(kf.dim_x, 4);
        assert_eq!(kf.dim_z, 2);
        assert_eq!(kf.x.len(), 4);
        assert_eq!(kf.p.nrows(), 4);
        assert_eq!(kf.p.ncols(), 4);

        // Verify F is identity
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(kf.f[(i, j)], expected, epsilon = 1e-10);
            }
        }

        // Verify H matrix is identity for measurement dimensions
        for i in 0..2 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(kf.h[(i, j)], expected, epsilon = 1e-10);
            }
        }

        // Verify initial state is zero
        for i in 0..4 {
            assert_relative_eq!(kf.x[i], 0.0, epsilon = 1e-10);
        }
    }

    // ===== Predict tests =====

    #[test]
    fn test_kalman_filter_predict() {
        let mut kf = KalmanFilter::new(2, 1);

        // Set initial state [position=1, velocity=2]
        kf.x = DVector::from_vec(vec![1.0, 2.0]);

        // Set F matrix for constant velocity model: [1 dt; 0 1]
        let dt = 1.0;
        kf.f = DMatrix::from_row_slice(2, 2, &[
            1.0, dt,
            0.0, 1.0,
        ]);

        // Set Q (process noise)
        kf.q = DMatrix::from_row_slice(2, 2, &[
            0.1, 0.0,
            0.0, 0.1,
        ]);

        // Initial covariance
        kf.p = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, 1.0,
        ]);

        // Predict
        kf.predict(None);

        // After prediction: x = F @ x = [1+2*1, 2] = [3, 2]
        assert_relative_eq!(kf.x[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(kf.x[1], 2.0, epsilon = 1e-10);

        // Covariance should increase: P = F @ P @ F' + Q
        // Expected: [2.1, 1; 1, 1.1]
        assert_relative_eq!(kf.p[(0, 0)], 2.1, epsilon = 1e-10);
        assert_relative_eq!(kf.p[(0, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(kf.p[(1, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(kf.p[(1, 1)], 1.1, epsilon = 1e-10);
    }

    // ===== Update tests =====

    #[test]
    fn test_kalman_filter_update() {
        let mut kf = KalmanFilter::new(2, 1);

        // Set initial state
        kf.x = DVector::from_vec(vec![0.0, 0.0]);

        // Set measurement matrix H = [1, 0] (measure position only)
        kf.h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

        // Set R (measurement noise)
        kf.r = DMatrix::from_row_slice(1, 1, &[1.0]);

        // Set P (large initial uncertainty)
        kf.p = DMatrix::from_row_slice(2, 2, &[
            10.0, 0.0,
            0.0, 10.0,
        ]);

        // Measurement: position = 5.0
        let z = DVector::from_vec(vec![5.0]);

        // Update
        kf.update(&z, None, None);

        // With large P and small R, estimate should move significantly toward measurement
        // Kalman gain K ≈ [10/(10+1), 0]' ≈ [0.909, 0]'
        // x = x + K @ (z - H @ x) = [0, 0] + [0.909, 0]' @ 5.0 ≈ [4.545, 0]
        assert_relative_eq!(kf.x[0], 4.545454545, epsilon = 1e-6);
        assert_relative_eq!(kf.x[1], 0.0, epsilon = 1e-10);
    }

    // ===== Predict-Update cycle tests =====

    #[test]
    fn test_kalman_filter_predict_update_cycle() {
        let mut kf = KalmanFilter::new(2, 1);

        // Simple 1D position+velocity tracking
        kf.x = DVector::from_vec(vec![0.0, 1.0]); // start at 0, moving at 1 unit/step

        // F matrix for dt=1: x_new = x + v, v_new = v
        kf.f = DMatrix::from_row_slice(2, 2, &[
            1.0, 1.0,
            0.0, 1.0,
        ]);

        // H matrix: measure position only
        kf.h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

        // Low noise
        kf.q = DMatrix::from_row_slice(2, 2, &[
            0.01, 0.0,
            0.0, 0.01,
        ]);
        kf.r = DMatrix::from_row_slice(1, 1, &[0.1]);
        kf.p = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.0,
            0.0, 1.0,
        ]);

        // Simulate movement: object moves from 0 to 5 in 5 steps
        let measurements = [1.0, 2.0, 3.0, 4.0, 5.0];

        for (i, &z_val) in measurements.iter().enumerate() {
            // Predict where object will be
            kf.predict(None);

            // Measure actual position
            let z = DVector::from_vec(vec![z_val]);
            kf.update(&z, None, None);

            // After a few iterations, state should track the linear motion
            if i >= 2 {
                // Position should be close to measurement
                let diff = (kf.x[0] - z_val).abs();
                assert!(diff < 0.5, "Step {}: position {} too far from measurement {}", i+1, kf.x[0], z_val);

                // Velocity should be close to 1.0
                let vel_diff = (kf.x[1] - 1.0).abs();
                assert!(vel_diff < 0.5, "Step {}: velocity {} should be close to 1.0", i+1, kf.x[1]);
            }
        }
    }

    // ===== Multi-dimensional tests =====

    #[test]
    fn test_kalman_filter_multi_dimensional() {
        // 3D position tracking: [x, y, z, vx, vy, vz]
        let dim_x = 6;
        let dim_z = 3;
        let mut kf = KalmanFilter::new(dim_x, dim_z);

        // Initial state
        kf.x = DVector::from_vec(vec![
            1.0, 2.0, 3.0, // positions
            0.5, 0.5, 0.5, // velocities
        ]);

        // Constant velocity model
        let dt = 1.0;
        kf.f = DMatrix::from_row_slice(dim_x, dim_x, &[
            1.0, 0.0, 0.0, dt,  0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, dt,  0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, dt,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);

        // Predict
        kf.predict(None);

        // After prediction with dt=1:
        // x_new = x + vx = 1.0 + 0.5 = 1.5
        // y_new = y + vy = 2.0 + 0.5 = 2.5
        // z_new = z + vz = 3.0 + 0.5 = 3.5
        // velocities unchanged
        assert_relative_eq!(kf.x[0], 1.5, epsilon = 1e-10);
        assert_relative_eq!(kf.x[1], 2.5, epsilon = 1e-10);
        assert_relative_eq!(kf.x[2], 3.5, epsilon = 1e-10);
        assert_relative_eq!(kf.x[3], 0.5, epsilon = 1e-10);
        assert_relative_eq!(kf.x[4], 0.5, epsilon = 1e-10);
        assert_relative_eq!(kf.x[5], 0.5, epsilon = 1e-10);
    }

    // ===== Getters/Setters tests =====

    #[test]
    fn test_kalman_filter_getters() {
        let mut kf = KalmanFilter::new(4, 2);

        // Set and get state
        let new_x = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        kf.x = new_x.clone();
        assert_eq!(kf.get_state(), &new_x);

        // Set and get covariance
        let mut new_p = DMatrix::zeros(4, 4);
        for i in 0..4 {
            new_p[(i, i)] = (i + 1) as f64;
        }
        kf.p = new_p.clone();
        assert_eq!(kf.get_covariance(), &new_p);
    }
}
