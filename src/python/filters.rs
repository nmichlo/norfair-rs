//! Python wrappers for filter factories.

use pyo3::prelude::*;

use crate::filter::{
    FilterFactory, FilterFactoryEnum, OptimizedKalmanFilterFactory, FilterPyKalmanFilterFactory, NoFilterFactory,
};

/// Wrapper enum for filter factories that can be used from Python.
/// This bridges Python filter factory types to the crate's FilterFactoryEnum.
pub enum PyFilterFactoryEnum {
    Optimized(OptimizedKalmanFilterFactory),
    FilterPy(FilterPyKalmanFilterFactory),
    NoFilter(NoFilterFactory),
}

impl PyFilterFactoryEnum {
    pub fn as_filter_factory(&self) -> &dyn FilterFactory {
        match self {
            PyFilterFactoryEnum::Optimized(f) => f,
            PyFilterFactoryEnum::FilterPy(f) => f,
            PyFilterFactoryEnum::NoFilter(f) => f,
        }
    }

    /// Convert to the crate's FilterFactoryEnum for use with Tracker.
    pub fn to_filter_factory_enum(&self) -> FilterFactoryEnum {
        match self {
            PyFilterFactoryEnum::Optimized(f) => FilterFactoryEnum::Optimized(f.clone()),
            PyFilterFactoryEnum::FilterPy(f) => FilterFactoryEnum::FilterPy(f.clone()),
            PyFilterFactoryEnum::NoFilter(f) => FilterFactoryEnum::None(f.clone()),
        }
    }
}

impl Clone for PyFilterFactoryEnum {
    fn clone(&self) -> Self {
        match self {
            PyFilterFactoryEnum::Optimized(f) => PyFilterFactoryEnum::Optimized(f.clone()),
            PyFilterFactoryEnum::FilterPy(f) => PyFilterFactoryEnum::FilterPy(f.clone()),
            PyFilterFactoryEnum::NoFilter(f) => PyFilterFactoryEnum::NoFilter(f.clone()),
        }
    }
}

/// Optimized Kalman filter factory with simplified covariance tracking.
///
/// Instead of tracking full covariance matrices, this filter tracks per-dimension
/// variances, making it faster for multi-point tracking while maintaining
/// reasonable accuracy.
///
/// This is the default filter factory used by Tracker.
#[pyclass(name = "OptimizedKalmanFilterFactory")]
#[derive(Clone)]
pub struct PyOptimizedKalmanFilterFactory {
    pub(crate) inner: OptimizedKalmanFilterFactory,
}

#[pymethods]
impl PyOptimizedKalmanFilterFactory {
    /// Create a new OptimizedKalmanFilterFactory.
    ///
    /// Args:
    ///     R: Measurement noise variance (default: 4.0).
    ///     Q: Process noise variance (default: 0.1).
    ///     pos_variance: Initial position variance (default: 10.0).
    ///     pos_vel_covariance: Initial position-velocity covariance (default: 0.0).
    ///     vel_variance: Initial velocity variance (default: 1.0).
    #[new]
    #[pyo3(signature = (R=4.0, Q=0.1, pos_variance=10.0, pos_vel_covariance=0.0, vel_variance=1.0))]
    fn new(R: f64, Q: f64, pos_variance: f64, pos_vel_covariance: f64, vel_variance: f64) -> Self {
        Self {
            inner: OptimizedKalmanFilterFactory::new(R, Q, pos_variance, pos_vel_covariance, vel_variance),
        }
    }

    fn __repr__(&self) -> String {
        "OptimizedKalmanFilterFactory()".to_string()
    }
}

impl PyOptimizedKalmanFilterFactory {
    pub fn to_enum(&self) -> PyFilterFactoryEnum {
        PyFilterFactoryEnum::Optimized(self.inner.clone())
    }
}

/// FilterPy-compatible Kalman filter factory.
///
/// This maintains full covariance matrices and provides behavior
/// equivalent to filterpy.kalman.KalmanFilter. More accurate but slower
/// than OptimizedKalmanFilterFactory for multi-point tracking.
#[pyclass(name = "FilterPyKalmanFilterFactory")]
#[derive(Clone)]
pub struct PyFilterPyKalmanFilterFactory {
    pub(crate) inner: FilterPyKalmanFilterFactory,
}

#[pymethods]
impl PyFilterPyKalmanFilterFactory {
    /// Create a new FilterPyKalmanFilterFactory.
    ///
    /// Args:
    ///     R: Measurement noise variance (default: 4.0).
    ///     Q: Process noise variance (default: 0.1).
    ///     P: Initial state covariance (default: 10.0).
    #[new]
    #[pyo3(signature = (R=4.0, Q=0.1, P=10.0))]
    fn new(R: f64, Q: f64, P: f64) -> Self {
        Self {
            inner: FilterPyKalmanFilterFactory::new(R, Q, P),
        }
    }

    fn __repr__(&self) -> String {
        "FilterPyKalmanFilterFactory()".to_string()
    }
}

impl PyFilterPyKalmanFilterFactory {
    pub fn to_enum(&self) -> PyFilterFactoryEnum {
        PyFilterFactoryEnum::FilterPy(self.inner.clone())
    }
}

/// No-filter factory (baseline without prediction).
///
/// This filter simply stores the last measurement without any prediction
/// or filtering. Useful for comparison or when Kalman filtering is not needed.
#[pyclass(name = "NoFilterFactory")]
#[derive(Clone)]
pub struct PyNoFilterFactory {
    pub(crate) inner: NoFilterFactory,
}

#[pymethods]
impl PyNoFilterFactory {
    /// Create a new NoFilterFactory.
    #[new]
    fn new() -> Self {
        Self {
            inner: NoFilterFactory::new(),
        }
    }

    fn __repr__(&self) -> String {
        "NoFilterFactory()".to_string()
    }
}

impl PyNoFilterFactory {
    pub fn to_enum(&self) -> PyFilterFactoryEnum {
        PyFilterFactoryEnum::NoFilter(self.inner.clone())
    }
}

/// Extract a FilterFactory from a Python object.
///
/// Supports:
/// - PyOptimizedKalmanFilterFactory
/// - PyFilterPyKalmanFilterFactory
/// - PyNoFilterFactory
/// - None (returns default OptimizedKalmanFilterFactory)
pub fn extract_filter_factory(obj: Option<&Bound<'_, PyAny>>) -> PyResult<PyFilterFactoryEnum> {
    match obj {
        None => Ok(PyFilterFactoryEnum::Optimized(OptimizedKalmanFilterFactory::default())),
        Some(py_obj) => {
            if let Ok(f) = py_obj.extract::<PyOptimizedKalmanFilterFactory>() {
                return Ok(f.to_enum());
            }
            if let Ok(f) = py_obj.extract::<PyFilterPyKalmanFilterFactory>() {
                return Ok(f.to_enum());
            }
            if let Ok(f) = py_obj.extract::<PyNoFilterFactory>() {
                return Ok(f.to_enum());
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "filter_factory must be an OptimizedKalmanFilterFactory, FilterPyKalmanFilterFactory, or NoFilterFactory"
            ))
        }
    }
}
