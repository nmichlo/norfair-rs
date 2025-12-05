//! Python bindings for norfair-rs using PyO3.
//!
//! This module provides Python bindings that match the Python norfair library API,
//! targeting compatibility with norfair v2.3.0.

use pyo3::prelude::*;

mod detection;
mod distances;
mod filters;
mod tracked_object;
mod tracker;
mod transforms;

pub use detection::PyDetection;
pub use distances::{
    get_distance_by_name, PyBuiltinDistance, PyDistanceFunctionWrapper, PyScalarDistance,
    PyVectorizedDistance,
};
pub use filters::{
    PyFilterPyKalmanFilterFactory, PyNoFilterFactory, PyOptimizedKalmanFilterFactory,
};
pub use tracked_object::PyTrackedObject;
pub use tracker::PyTracker;
pub use transforms::PyTranslationTransformation;

/// Python module for norfair-rs.
///
/// Provides object tracking functionality compatible with the Python norfair library.
/// The function is named `_norfair_rs` with underscore prefix for mixed Python/Rust projects.
#[pymodule]
fn _norfair_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core classes
    m.add_class::<PyDetection>()?;
    m.add_class::<PyTrackedObject>()?;
    m.add_class::<PyTracker>()?;

    // Filter factories
    m.add_class::<PyOptimizedKalmanFilterFactory>()?;
    m.add_class::<PyFilterPyKalmanFilterFactory>()?;
    m.add_class::<PyNoFilterFactory>()?;

    // Distance classes
    m.add_class::<PyScalarDistance>()?;
    m.add_class::<PyVectorizedDistance>()?;
    m.add_class::<PyBuiltinDistance>()?;
    m.add_class::<PyDistanceFunctionWrapper>()?;

    // Transformations
    m.add_class::<PyTranslationTransformation>()?;

    // Distance functions
    m.add_function(wrap_pyfunction!(distances::get_distance_by_name, m)?)?;
    m.add_function(wrap_pyfunction!(distances::frobenius, m)?)?;
    m.add_function(wrap_pyfunction!(distances::mean_euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(distances::mean_manhattan, m)?)?;
    m.add_function(wrap_pyfunction!(distances::iou, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__norfair_compat_version__", "2.3.0")?;

    Ok(())
}
