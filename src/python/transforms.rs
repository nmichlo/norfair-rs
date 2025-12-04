//! Python wrappers for coordinate transformations.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use crate::camera_motion::{
    CoordinateTransformation, TranslationTransformation, NilCoordinateTransformation,
};

use super::detection::{numpy_to_dmatrix, dmatrix_to_numpy};

/// Enum to hold different transformation implementations
pub enum PyTransformEnum {
    Translation(TranslationTransformation),
    Nil(NilCoordinateTransformation),
}

impl PyTransformEnum {
    pub fn as_transform(&self) -> &dyn CoordinateTransformation {
        match self {
            PyTransformEnum::Translation(t) => t,
            PyTransformEnum::Nil(t) => t,
        }
    }
}

/// Simple 2D translation transformation (camera pan/tilt without rotation/zoom).
///
/// This transformation adds/subtracts a movement vector to convert between
/// relative (camera frame) and absolute (world frame) coordinates.
#[pyclass(name = "TranslationTransformation")]
#[derive(Clone)]
pub struct PyTranslationTransformation {
    pub(crate) inner: TranslationTransformation,
}

#[pymethods]
impl PyTranslationTransformation {
    /// Create a new TranslationTransformation.
    ///
    /// Args:
    ///     movement_vector: A 2-element array [dx, dy] representing the camera movement.
    #[new]
    fn new(movement_vector: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let arr = movement_vector.as_array();
        if arr.len() != 2 {
            return Err(PyValueError::new_err(
                "movement_vector must have exactly 2 elements [dx, dy]"
            ));
        }

        Ok(Self {
            inner: TranslationTransformation::new([arr[0], arr[1]]),
        })
    }

    /// Transform points from absolute (world frame) to relative (camera frame) coordinates.
    ///
    /// Args:
    ///     points: Array of points, shape (n_points, 2).
    ///
    /// Returns:
    ///     Transformed points in relative coordinates.
    fn abs_to_rel<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pts = numpy_to_dmatrix(&points);
        let result = self.inner.abs_to_rel(&pts);
        Ok(dmatrix_to_numpy(py, &result))
    }

    /// Transform points from relative (camera frame) to absolute (world frame) coordinates.
    ///
    /// Args:
    ///     points: Array of points, shape (n_points, 2).
    ///
    /// Returns:
    ///     Transformed points in absolute coordinates.
    fn rel_to_abs<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pts = numpy_to_dmatrix(&points);
        let result = self.inner.rel_to_abs(&pts);
        Ok(dmatrix_to_numpy(py, &result))
    }

    /// The movement vector [dx, dy].
    #[getter]
    fn movement_vector(&self) -> [f64; 2] {
        self.inner.movement_vector
    }

    fn __repr__(&self) -> String {
        format!(
            "TranslationTransformation([{:.2}, {:.2}])",
            self.inner.movement_vector[0],
            self.inner.movement_vector[1]
        )
    }
}

impl PyTranslationTransformation {
    pub fn to_enum(&self) -> PyTransformEnum {
        PyTransformEnum::Translation(self.inner.clone())
    }
}

/// Extract a coordinate transformation from a Python object.
///
/// Supports:
/// - PyTranslationTransformation
/// - None (returns None, meaning no transformation)
pub fn extract_transform(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PyTransformEnum>> {
    match obj {
        None => Ok(None),
        Some(py_obj) => {
            if py_obj.is_none() {
                return Ok(None);
            }
            if let Ok(t) = py_obj.extract::<PyTranslationTransformation>() {
                return Ok(Some(t.to_enum()));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "coord_transformations must be a TranslationTransformation or None"
            ))
        }
    }
}
