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

/// Build a `types.GenericAlias` so a `#[pyclass]` can be subscripted at runtime.
///
/// `Detection`, `TrackedObject` and `Tracker` are generic in the type stubs,
/// parameterized by the type of the user payload carried in `Detection.data`.
/// PyO3 classes are not subscriptable by default, so `Detection[MyPayload]`
/// would raise `TypeError` even though the annotation is valid.
pub(crate) fn generic_alias(
    cls: &Bound<'_, pyo3::types::PyType>,
    item: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let types = cls.py().import("types")?;
    Ok(types.getattr("GenericAlias")?.call1((cls, item))?.unbind())
}

/// Reset global ID counter (for testing only).
///
/// This resets the global ID counter used for TrackedObject global IDs.
/// Only use this in test fixtures to ensure consistent IDs across test runs.
#[pyfunction]
fn _reset_global_id_counter() {
    crate::tracked_object::reset_global_counter();
}

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

    // Test utilities
    m.add_function(wrap_pyfunction!(_reset_global_id_counter, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__norfair_compat_version__", "2.3.0")?;

    Ok(())
}
