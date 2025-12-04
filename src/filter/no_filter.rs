//! No-op filter that simply stores the last measurement.
//!
//! This provides a baseline for comparison - no prediction, just tracking
//! the last known position.

use nalgebra::{DMatrix, DVector};
use super::traits::{Filter, FilterFactory};

/// No-op filter that stores the last measurement.
///
/// This filter does not perform any prediction or filtering.
/// It simply stores the last measurement as the current state.
#[derive(Clone, Debug)]
pub struct NoFilter {
    x: DVector<f64>,
    dim_z: usize,
    dim_x: usize,
    n_points: usize,
    n_dims: usize,
}

impl NoFilter {
    /// Create a new NoFilter initialized with the detection.
    pub fn new(initial_detection: &DMatrix<f64>) -> Self {
        let n_points = initial_detection.nrows();
        let n_dims = initial_detection.ncols();
        let dim_z = n_points * n_dims;
        let dim_x = dim_z * 2; // Keep same structure as other filters

        // Initialize state: [positions..., velocities...]
        let mut x = DVector::zeros(dim_x);
        for i in 0..n_points {
            for j in 0..n_dims {
                x[i * n_dims + j] = initial_detection[(i, j)];
            }
        }

        Self {
            x,
            dim_z,
            dim_x,
            n_points,
            n_dims,
        }
    }
}

impl Filter for NoFilter {
    fn predict(&mut self) {
        // No-op: don't predict anything
    }

    fn update(
        &mut self,
        measurement: &DVector<f64>,
        _r: Option<&DMatrix<f64>>,
        _h: Option<&DMatrix<f64>>,
    ) {
        // Simply copy the measurement to the position part of state
        for i in 0..self.dim_z.min(measurement.len()) {
            self.x[i] = measurement[i];
        }
    }

    fn get_state(&self) -> DMatrix<f64> {
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

/// Factory for creating NoFilter instances.
#[derive(Clone, Debug, Default)]
pub struct NoFilterFactory;

impl NoFilterFactory {
    pub fn new() -> Self {
        Self
    }
}

impl FilterFactory for NoFilterFactory {
    fn create_filter(&self, initial_detection: &DMatrix<f64>) -> Box<dyn Filter> {
        Box::new(NoFilter::new(initial_detection))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_no_filter_create() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = NoFilterFactory::new();
        let filter = factory.create_filter(&initial);

        assert_eq!(filter.dim_z(), 2);
        assert_eq!(filter.dim_x(), 4);

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(state[(0, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_no_filter_predict_is_noop() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = NoFilterFactory::new();
        let mut filter = factory.create_filter(&initial);

        let state_before = filter.get_state();
        filter.predict();
        let state_after = filter.get_state();

        assert_relative_eq!(state_after[(0, 0)], state_before[(0, 0)], epsilon = 1e-10);
        assert_relative_eq!(state_after[(0, 1)], state_before[(0, 1)], epsilon = 1e-10);
    }

    #[test]
    fn test_no_filter_update() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = NoFilterFactory::new();
        let mut filter = factory.create_filter(&initial);

        let measurement = DVector::from_vec(vec![2.0, 3.0]);
        filter.update(&measurement, None, None);

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(state[(0, 1)], 3.0, epsilon = 1e-10);
    }
}
