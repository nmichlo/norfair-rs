//! Enum-based filter dispatch for static (non-virtual) function calls.
//!
//! This module provides `FilterEnum` and `FilterFactoryEnum` that wrap all supported
//! filter types and dispatch without vtable lookups, improving performance
//! for hot-path code.

use nalgebra::{DMatrix, DVector};
use super::traits::{Filter, FilterFactory};
use super::optimized::{OptimizedKalmanFilter, OptimizedKalmanFilterFactory};
use super::filterpy::{FilterPyKalmanFilter, FilterPyKalmanFilterFactory};
use super::no_filter::{NoFilter, NoFilterFactory};

/// Enum-based filter for static dispatch.
///
/// This avoids `Box<dyn Filter>` vtable overhead by using an enum
/// with inline implementations.
#[derive(Clone, Debug)]
pub enum FilterEnum {
    Optimized(OptimizedKalmanFilter),
    FilterPy(FilterPyKalmanFilter),
    None(NoFilter),
}

impl FilterEnum {
    #[inline(always)]
    pub fn predict(&mut self) {
        match self {
            FilterEnum::Optimized(f) => f.predict(),
            FilterEnum::FilterPy(f) => f.predict(),
            FilterEnum::None(f) => f.predict(),
        }
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        measurement: &DVector<f64>,
        r: Option<&DMatrix<f64>>,
        h: Option<&DMatrix<f64>>,
    ) {
        match self {
            FilterEnum::Optimized(f) => f.update(measurement, r, h),
            FilterEnum::FilterPy(f) => f.update(measurement, r, h),
            FilterEnum::None(f) => f.update(measurement, r, h),
        }
    }

    #[inline(always)]
    pub fn get_state(&self) -> DMatrix<f64> {
        match self {
            FilterEnum::Optimized(f) => f.get_state(),
            FilterEnum::FilterPy(f) => f.get_state(),
            FilterEnum::None(f) => f.get_state(),
        }
    }

    #[inline(always)]
    pub fn get_state_vector(&self) -> &DVector<f64> {
        match self {
            FilterEnum::Optimized(f) => f.get_state_vector(),
            FilterEnum::FilterPy(f) => f.get_state_vector(),
            FilterEnum::None(f) => f.get_state_vector(),
        }
    }

    #[inline(always)]
    pub fn set_state_vector(&mut self, x: &DVector<f64>) {
        match self {
            FilterEnum::Optimized(f) => f.set_state_vector(x),
            FilterEnum::FilterPy(f) => f.set_state_vector(x),
            FilterEnum::None(f) => f.set_state_vector(x),
        }
    }

    #[inline(always)]
    pub fn dim_z(&self) -> usize {
        match self {
            FilterEnum::Optimized(f) => f.dim_z(),
            FilterEnum::FilterPy(f) => f.dim_z(),
            FilterEnum::None(f) => f.dim_z(),
        }
    }

    #[inline(always)]
    pub fn dim_x(&self) -> usize {
        match self {
            FilterEnum::Optimized(f) => f.dim_x(),
            FilterEnum::FilterPy(f) => f.dim_x(),
            FilterEnum::None(f) => f.dim_x(),
        }
    }
}

// Implement the Filter trait for FilterEnum so it can be used with existing code
impl Filter for FilterEnum {
    #[inline(always)]
    fn predict(&mut self) {
        FilterEnum::predict(self)
    }

    #[inline(always)]
    fn update(
        &mut self,
        measurement: &DVector<f64>,
        r: Option<&DMatrix<f64>>,
        h: Option<&DMatrix<f64>>,
    ) {
        FilterEnum::update(self, measurement, r, h)
    }

    #[inline(always)]
    fn get_state(&self) -> DMatrix<f64> {
        FilterEnum::get_state(self)
    }

    #[inline(always)]
    fn get_state_vector(&self) -> &DVector<f64> {
        FilterEnum::get_state_vector(self)
    }

    #[inline(always)]
    fn set_state_vector(&mut self, x: &DVector<f64>) {
        FilterEnum::set_state_vector(self, x)
    }

    #[inline(always)]
    fn dim_z(&self) -> usize {
        FilterEnum::dim_z(self)
    }

    #[inline(always)]
    fn dim_x(&self) -> usize {
        FilterEnum::dim_x(self)
    }
}

/// Enum-based filter factory for static dispatch.
///
/// This avoids `Box<dyn FilterFactory>` vtable overhead by using an enum
/// with inline implementations.
#[derive(Clone, Debug)]
pub enum FilterFactoryEnum {
    Optimized(OptimizedKalmanFilterFactory),
    FilterPy(FilterPyKalmanFilterFactory),
    None(NoFilterFactory),
}

impl Default for FilterFactoryEnum {
    fn default() -> Self {
        FilterFactoryEnum::Optimized(OptimizedKalmanFilterFactory::default())
    }
}

impl FilterFactoryEnum {
    /// Create a new filter with static dispatch.
    #[inline(always)]
    pub fn create(&self, initial_detection: &DMatrix<f64>) -> FilterEnum {
        match self {
            FilterFactoryEnum::Optimized(f) => {
                FilterEnum::Optimized(OptimizedKalmanFilter::new(
                    initial_detection,
                    f.r(),
                    f.q(),
                    f.pos_variance(),
                    f.pos_vel_covariance(),
                    f.vel_variance(),
                ))
            }
            FilterFactoryEnum::FilterPy(f) => {
                FilterEnum::FilterPy(FilterPyKalmanFilter::new(
                    initial_detection,
                    f.r(),
                    f.q(),
                    f.p(),
                ))
            }
            FilterFactoryEnum::None(_) => {
                FilterEnum::None(NoFilter::new(initial_detection))
            }
        }
    }
}

// Implement the FilterFactory trait for compatibility with existing code
impl FilterFactory for FilterFactoryEnum {
    fn create_filter(&self, initial_detection: &DMatrix<f64>) -> Box<dyn Filter> {
        // This path still uses Box for compatibility, but the enum can be used directly
        Box::new(self.create(initial_detection))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_filter_enum_optimized() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = FilterFactoryEnum::Optimized(OptimizedKalmanFilterFactory::default());
        let mut filter = factory.create(&initial);

        assert_eq!(filter.dim_z(), 2);
        assert_eq!(filter.dim_x(), 4);

        filter.predict();
        let measurement = DVector::from_vec(vec![1.0, 1.0]);
        filter.update(&measurement, None, None);

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_filter_enum_filterpy() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = FilterFactoryEnum::FilterPy(FilterPyKalmanFilterFactory::default());
        let mut filter = factory.create(&initial);

        assert_eq!(filter.dim_z(), 2);
        assert_eq!(filter.dim_x(), 4);

        filter.predict();
        let measurement = DVector::from_vec(vec![1.0, 1.0]);
        filter.update(&measurement, None, None);

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_filter_enum_none() {
        let initial = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let factory = FilterFactoryEnum::None(NoFilterFactory::new());
        let mut filter = factory.create(&initial);

        assert_eq!(filter.dim_z(), 2);

        let measurement = DVector::from_vec(vec![2.0, 3.0]);
        filter.update(&measurement, None, None);

        let state = filter.get_state();
        assert_relative_eq!(state[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(state[(0, 1)], 3.0, epsilon = 1e-10);
    }
}
