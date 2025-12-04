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
