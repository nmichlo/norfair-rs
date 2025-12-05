//! Python wrappers for coordinate transformations.

use nalgebra::DMatrix;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::camera_motion::{
    CoordinateTransformation, NilCoordinateTransformation, TranslationTransformation,
};

use super::detection::{dmatrix_to_numpy, numpy_to_dmatrix};

/// A wrapper for duck-typed Python coordinate transformations.
///
/// This wraps any Python object that has `abs_to_rel` and `rel_to_abs` methods.
#[derive(Debug)]
pub struct PyCallableTransformation {
    /// The Python object with abs_to_rel and rel_to_abs methods
    py_obj: Py<PyAny>,
}

impl PyCallableTransformation {
    pub fn new(py_obj: Py<PyAny>) -> Self {
        Self { py_obj }
    }
}

// Py<PyAny> is Send (but not Sync by default), but we ensure thread safety
// by always acquiring the GIL before accessing the Python object.
unsafe impl Send for PyCallableTransformation {}
unsafe impl Sync for PyCallableTransformation {}

impl CoordinateTransformation for PyCallableTransformation {
    fn rel_to_abs(&self, points: &DMatrix<f64>) -> DMatrix<f64> {
        Python::with_gil(|py| {
            let np_points = dmatrix_to_numpy(py, points);
            match self.py_obj.call_method1(py, "rel_to_abs", (np_points,)) {
                Ok(result) => {
                    match numpy_to_dmatrix(py, result.bind(py)) {
                        Ok(mat) => mat,
                        Err(_) => points.clone(), // Fallback on error
                    }
                }
                Err(_) => points.clone(), // Fallback on error
            }
        })
    }

    fn abs_to_rel(&self, points: &DMatrix<f64>) -> DMatrix<f64> {
        Python::with_gil(|py| {
            let np_points = dmatrix_to_numpy(py, points);
            match self.py_obj.call_method1(py, "abs_to_rel", (np_points,)) {
                Ok(result) => {
                    match numpy_to_dmatrix(py, result.bind(py)) {
                        Ok(mat) => mat,
                        Err(_) => points.clone(), // Fallback on error
                    }
                }
                Err(_) => points.clone(), // Fallback on error
            }
        })
    }

    fn clone_box(&self) -> Box<dyn CoordinateTransformation> {
        Box::new(PyCallableTransformation {
            py_obj: Python::with_gil(|py| self.py_obj.clone_ref(py)),
        })
    }
}

/// Enum to hold different transformation implementations
pub enum PyTransformEnum {
    Translation(TranslationTransformation),
    Nil(NilCoordinateTransformation),
    PyCallable(PyCallableTransformation),
}

impl PyTransformEnum {
    pub fn as_transform(&self) -> &dyn CoordinateTransformation {
        match self {
            PyTransformEnum::Translation(t) => t,
            PyTransformEnum::Nil(t) => t,
            PyTransformEnum::PyCallable(t) => t,
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
    fn new(py: Python<'_>, movement_vector: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        let np = py.import("numpy")?;
        let arr_f64 = np
            .call_method1("asarray", (movement_vector,))?
            .call_method1("astype", (np.getattr("float64")?,))?
            .call_method0("ravel")?;
        let arr: Bound<'_, PyArray1<f64>> = arr_f64.extract()?;
        let arr_readonly = arr.readonly();
        let view = arr_readonly.as_array();

        if view.len() != 2 {
            return Err(PyValueError::new_err(
                "movement_vector must have exactly 2 elements [dx, dy]",
            ));
        }

        Ok(Self {
            inner: TranslationTransformation::new([view[0], view[1]]),
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
        points: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pts = numpy_to_dmatrix(py, points)?;
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
        points: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pts = numpy_to_dmatrix(py, points)?;
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
            self.inner.movement_vector[0], self.inner.movement_vector[1]
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
/// - PyTranslationTransformation (native norfair_rs type)
/// - Any Python object with abs_to_rel and rel_to_abs methods (duck-typed)
/// - None (returns None, meaning no transformation)
pub fn extract_transform(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<PyTransformEnum>> {
    match obj {
        None => Ok(None),
        Some(py_obj) => {
            if py_obj.is_none() {
                return Ok(None);
            }
            // Try native TranslationTransformation first
            if let Ok(t) = py_obj.extract::<PyTranslationTransformation>() {
                return Ok(Some(t.to_enum()));
            }
            // Try duck-typed: any object with abs_to_rel and rel_to_abs methods
            let has_abs_to_rel = py_obj.hasattr("abs_to_rel").unwrap_or(false);
            let has_rel_to_abs = py_obj.hasattr("rel_to_abs").unwrap_or(false);
            if has_abs_to_rel && has_rel_to_abs {
                return Ok(Some(PyTransformEnum::PyCallable(
                    PyCallableTransformation::new(py_obj.clone().unbind()),
                )));
            }
            Err(pyo3::exceptions::PyTypeError::new_err(
                "coord_transformations must have abs_to_rel and rel_to_abs methods, or be None",
            ))
        }
    }
}
