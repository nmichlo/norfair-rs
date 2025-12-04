//! Filter traits for the tracking system.

use nalgebra::{DMatrix, DVector};

/// Trait for Kalman filter implementations.
///
/// All filter implementations must support predict and update operations,
/// as well as state access methods.
pub trait Filter: Send + Sync {
    /// Predict the next state without a measurement.
    fn predict(&mut self);

    /// Update the state with a measurement.
    ///
    /// # Arguments
    /// * `measurement` - Measurement vector (dim_z x 1)
    /// * `r` - Optional measurement noise covariance (overrides default)
    /// * `h` - Optional measurement matrix (overrides default, for partial observations)
    fn update(
        &mut self,
        measurement: &DVector<f64>,
        r: Option<&DMatrix<f64>>,
        h: Option<&DMatrix<f64>>,
    );

    /// Get the current state estimate (position only, reshaped to n_points x n_dims).
    fn get_state(&self) -> DMatrix<f64>;

    /// Get the full state vector (position + velocity).
    fn get_state_vector(&self) -> &DVector<f64>;

    /// Set the full state vector.
    fn set_state_vector(&mut self, x: &DVector<f64>);

    /// Get the measurement dimension.
    fn dim_z(&self) -> usize;

    /// Get the state dimension.
    fn dim_x(&self) -> usize;
}

/// Factory for creating filter instances.
///
/// This allows trackers to create new filters for each tracked object
/// without knowing the specific filter implementation.
pub trait FilterFactory: Send + Sync {
    /// Create a new filter initialized with the given detection.
    ///
    /// # Arguments
    /// * `initial_detection` - Initial detection points (n_points x n_dims)
    ///
    /// # Returns
    /// A new filter instance initialized with the detection position
    /// and zero velocity.
    fn create_filter(&self, initial_detection: &DMatrix<f64>) -> Box<dyn Filter>;
}
