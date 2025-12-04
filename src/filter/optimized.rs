//! Optimized Kalman filter with simplified covariance tracking.
//!
//! This filter tracks variance per dimension rather than full covariance matrices,
//! making it faster for multi-point tracking.

use nalgebra::{DMatrix, DVector};
use super::traits::{Filter, FilterFactory};

/// Optimized Kalman filter with simplified covariance.
///
/// Instead of tracking full covariance matrices, this filter tracks:
/// - Position variance per dimension
/// - Velocity variance per dimension
/// - Position-velocity covariance per dimension
///
/// This is much faster for tracking many points while maintaining
/// reasonable accuracy.
#[derive(Clone, Debug)]
pub struct OptimizedKalmanFilter {
    /// State vector [positions..., velocities...]
    x: DVector<f64>,
    /// Position variance per measurement dimension
    pos_variance: Vec<f64>,
    /// Velocity variance per measurement dimension
    vel_variance: Vec<f64>,
    /// Position-velocity covariance per measurement dimension
    pos_vel_covariance: Vec<f64>,
    /// Measurement noise variance
    r: f64,
    /// Process noise variance
    q: f64,
    /// Measurement dimension (n_points * n_dims)
    dim_z: usize,
    /// State dimension (2 * dim_z for position + velocity)
    dim_x: usize,
    /// Number of points
    n_points: usize,
    /// Point dimensionality
    n_dims: usize,
}

impl OptimizedKalmanFilter {
    /// Create a new optimized Kalman filter.
    pub fn new(
        initial_detection: &DMatrix<f64>,
        r: f64,
        q: f64,
        pos_variance: f64,
        pos_vel_covariance: f64,
        vel_variance: f64,
    ) -> Self {
        let n_points = initial_detection.nrows();
        let n_dims = initial_detection.ncols();
        let dim_z = n_points * n_dims;
        let dim_x = dim_z * 2;

        // Initialize state: [positions, velocities]
        let mut x = DVector::zeros(dim_x);
        // Copy positions
        for i in 0..n_points {
            for j in 0..n_dims {
                x[i * n_dims + j] = initial_detection[(i, j)];
            }
        }
        // Velocities start at 0

        // Initialize variance vectors
        let pos_variance_vec = vec![pos_variance; dim_z];
        let vel_variance_vec = vec![vel_variance; dim_z];
        let pos_vel_covariance_vec = vec![pos_vel_covariance; dim_z];

        Self {
            x,
            pos_variance: pos_variance_vec,
            vel_variance: vel_variance_vec,
            pos_vel_covariance: pos_vel_covariance_vec,
            r,
            q,
            dim_z,
            dim_x,
            n_points,
            n_dims,
        }
    }
}

impl Filter for OptimizedKalmanFilter {
    fn predict(&mut self) {
        // Update positions: pos += vel
        // NOTE: Covariance is NOT updated here - it's handled inline in update()
        // This matches Go's OptimizedKalmanFilter behavior
        for i in 0..self.dim_z {
            self.x[i] += self.x[self.dim_z + i];
        }
    }

    fn update(
        &mut self,
        measurement: &DVector<f64>,
        _r: Option<&DMatrix<f64>>,
        h: Option<&DMatrix<f64>>,
    ) {
        // Build observation mask from H matrix if provided
        // diagonal = 1.0 if observed, 0.0 if not
        let (diagonal, one_minus_diagonal): (Vec<f64>, Vec<f64>) = if let Some(h_mat) = h {
            let d: Vec<f64> = (0..self.dim_z).map(|i| h_mat[(i, i)]).collect();
            let omd: Vec<f64> = d.iter().map(|&x| 1.0 - x).collect();
            (d, omd)
        } else {
            (vec![1.0; self.dim_z], vec![0.0; self.dim_z])
        };

        // Precompute terms that include covariance propagation (matches Go behavior)
        // This embeds the predict-step covariance update into the update formula
        let mut vel_var_plus_pos_vel_cov = vec![0.0; self.dim_z];
        let mut added_variances = vec![0.0; self.dim_z];
        let mut kalman_r_over_added_variances = vec![0.0; self.dim_z];
        let mut vel_var_plus_pos_vel_cov_over_added_variances = vec![0.0; self.dim_z];
        let mut added_variances_or_kalman_r = vec![0.0; self.dim_z];

        for i in 0..self.dim_z {
            vel_var_plus_pos_vel_cov[i] = self.pos_vel_covariance[i] + self.vel_variance[i];
            // added_variances = predicted_pos_var + R
            // where predicted_pos_var = pos_var + 2*pos_vel_cov + vel_var + q
            added_variances[i] = self.pos_variance[i]
                + self.pos_vel_covariance[i]
                + vel_var_plus_pos_vel_cov[i]
                + self.q
                + self.r;
            kalman_r_over_added_variances[i] = self.r / added_variances[i];
            vel_var_plus_pos_vel_cov_over_added_variances[i] =
                vel_var_plus_pos_vel_cov[i] / added_variances[i];
            added_variances_or_kalman_r[i] =
                added_variances[i] * one_minus_diagonal[i] + self.r * diagonal[i];
        }

        // Compute error (innovation)
        let mut error = vec![0.0; self.dim_z];
        for i in 0..self.dim_z {
            error[i] = (measurement[i] - self.x[i]) * diagonal[i];
        }

        // State update
        for i in 0..self.dim_z {
            // Position update: x += diagonal * (1 - R/addedVar) * error
            self.x[i] += diagonal[i] * (1.0 - kalman_r_over_added_variances[i]) * error[i];
        }
        for i in 0..self.dim_z {
            // Velocity update: v += diagonal * k_vel * error
            self.x[self.dim_z + i] +=
                diagonal[i] * vel_var_plus_pos_vel_cov_over_added_variances[i] * error[i];
        }

        // Covariance update
        for i in 0..self.dim_z {
            self.pos_variance[i] =
                (1.0 - kalman_r_over_added_variances[i]) * added_variances_or_kalman_r[i];
            self.pos_vel_covariance[i] =
                vel_var_plus_pos_vel_cov_over_added_variances[i] * added_variances_or_kalman_r[i];
            self.vel_variance[i] += self.q
                - diagonal[i]
                * vel_var_plus_pos_vel_cov_over_added_variances[i]
                * vel_var_plus_pos_vel_cov_over_added_variances[i]
                * added_variances[i];
        }
    }

    fn get_state(&self) -> DMatrix<f64> {
        // Return positions reshaped to (n_points, n_dims)
        let mut result = DMatrix::zeros(self.n_points, self.n_dims);
        for i in 0..self.n_points {
            for j in 0..self.n_dims {
                result[(i, j)] = self.x[i * self.n_dims + j];
            }
        }
        result
    }

    fn get_state_vector(&self) -> &DVector<f64> {
        &self.x
    }

    fn set_state_vector(&mut self, x: &DVector<f64>) {
        self.x.copy_from(x);
    }

    fn dim_z(&self) -> usize {
        self.dim_z
    }

    fn dim_x(&self) -> usize {
        self.dim_x
    }
}

/// Factory for creating OptimizedKalmanFilter instances.
#[derive(Clone, Debug)]
pub struct OptimizedKalmanFilterFactory {
    r: f64,
    q: f64,
    pos_variance: f64,
    pos_vel_covariance: f64,
    vel_variance: f64,
}

impl OptimizedKalmanFilterFactory {
    /// Create a new factory with the specified parameters.
    ///
    /// # Arguments
    /// * `r` - Measurement noise variance
    /// * `q` - Process noise variance
    /// * `pos_variance` - Initial position variance
    /// * `pos_vel_covariance` - Initial position-velocity covariance
    /// * `vel_variance` - Initial velocity variance
    pub fn new(
        r: f64,
        q: f64,
        pos_variance: f64,
        pos_vel_covariance: f64,
        vel_variance: f64,
    ) -> Self {
        Self {
            r,
            q,
            pos_variance,
            pos_vel_covariance,
            vel_variance,
        }
    }
}

impl Default for OptimizedKalmanFilterFactory {
    fn default() -> Self {
        Self::new(4.0, 0.1, 10.0, 0.0, 1.0)
    }
}

impl FilterFactory for OptimizedKalmanFilterFactory {
    fn create_filter(&self, initial_detection: &DMatrix<f64>) -> Box<dyn Filter> {
        Box::new(OptimizedKalmanFilter::new(
            initial_detection,
            self.r,
            self.q,
            self.pos_variance,
            self.pos_vel_covariance,
            self.vel_variance,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_optimized_kalman_create() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = OptimizedKalmanFilterFactory::default();
        let filter = factory.create_filter(&initial);

        assert_eq!(filter.dim_z(), 2);
        assert_eq!(filter.dim_x(), 4);

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(state[(0, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_optimized_kalman_static_object() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = OptimizedKalmanFilterFactory::default();
        let mut filter = factory.create_filter(&initial);

        // Run several predict-update cycles with static measurement
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
    fn test_optimized_kalman_moving_object() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = OptimizedKalmanFilterFactory::default();
        let mut filter = factory.create_filter(&initial);

        // Object moving in y direction
        let positions = vec![1.0, 2.0, 3.0, 4.0];
        for y in positions {
            let measurement = DVector::from_vec(vec![1.0, y]);
            filter.update(&measurement, None, None);
            filter.predict();
        }

        let state = filter.get_state();
        // Position should be close to last measurement
        assert_relative_eq!(state[(0, 0)], 1.0, epsilon = 0.5);
        assert!(state[(0, 1)] > 3.0 && state[(0, 1)] <= 5.0);
    }
}
