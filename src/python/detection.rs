//! Python wrapper for Detection.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use numpy::ndarray::{Array1, Array2};
use nalgebra::DMatrix;
use std::sync::{Arc, RwLock};

use crate::Detection;

/// A detection to be tracked.
///
/// Represents a detected object in a frame, with its position points
/// and optional metadata like scores, labels, and embeddings.
///
/// Compatible with norfair.drawing functions via duck-typing.
#[pyclass(name = "Detection")]
#[derive(Clone)]
pub struct PyDetection {
    pub(crate) inner: Arc<RwLock<Detection>>,
}

impl PyDetection {
    /// Create a new PyDetection wrapping a Rust Detection.
    pub fn from_detection(det: Detection) -> Self {
        Self {
            inner: Arc::new(RwLock::new(det)),
        }
    }

    /// Get a clone of the inner Detection.
    pub fn get_detection(&self) -> Detection {
        self.inner.read().unwrap().clone()
    }
}

#[pymethods]
impl PyDetection {
    /// Create a new Detection.
    ///
    /// Args:
    ///     points: Detection points as a numpy array of shape (n_points, n_dims).
    ///             For keypoints: [[x1, y1], [x2, y2], ...]
    ///             For bounding boxes: [[x1, y1], [x2, y2]] (top-left, bottom-right)
    ///     scores: Optional per-point confidence scores of shape (n_points,).
    ///     data: Optional arbitrary user data.
    ///     label: Optional class label for multi-class tracking.
    ///     embedding: Optional embedding vector for re-identification.
    #[new]
    #[pyo3(signature = (points, scores=None, data=None, label=None, embedding=None))]
    #[allow(unused_variables)]
    fn new(
        py: Python<'_>,
        points: PyReadonlyArray2<f64>,
        scores: Option<PyReadonlyArray1<f64>>,
        data: Option<PyObject>,
        label: Option<String>,
        embedding: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        // Convert numpy array to DMatrix (row-major)
        let points_arr = points.as_array();
        let n_points = points_arr.nrows();
        let n_dims = points_arr.ncols();

        // Validate dimensions
        if n_dims != 2 && n_dims != 3 {
            return Err(PyValueError::new_err(format!(
                "Points must have 2 or 3 dimensions, got {}",
                n_dims
            )));
        }

        // Convert to row-major Vec for DMatrix
        let mut data_vec = Vec::with_capacity(n_points * n_dims);
        for i in 0..n_points {
            for j in 0..n_dims {
                data_vec.push(points_arr[[i, j]]);
            }
        }
        let points_matrix = DMatrix::from_row_slice(n_points, n_dims, &data_vec);

        // Convert scores if provided
        let scores_vec = scores.map(|s| {
            let arr = s.as_array();
            arr.iter().cloned().collect::<Vec<f64>>()
        });

        // Validate scores length
        if let Some(ref sv) = scores_vec {
            if sv.len() != n_points {
                return Err(PyValueError::new_err(format!(
                    "Scores length {} doesn't match {} points",
                    sv.len(),
                    n_points
                )));
            }
        }

        // Convert embedding if provided
        let embedding_vec = embedding.map(|e| {
            let arr = e.as_array();
            arr.iter().cloned().collect::<Vec<f64>>()
        });

        // Create Detection
        let det = Detection::with_config(points_matrix, scores_vec, label, embedding_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self::from_detection(det))
    }

    /// The detection points as a numpy array of shape (n_points, n_dims).
    #[getter]
    fn points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let det = self.inner.read().unwrap();
        Ok(dmatrix_to_numpy(py, &det.points))
    }

    /// Optional per-point confidence scores.
    #[getter]
    fn scores<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        let det = self.inner.read().unwrap();
        det.scores
            .as_ref()
            .map(|s| vec_to_numpy1(py, s))
    }

    /// Optional class label.
    #[getter]
    fn label(&self) -> Option<String> {
        self.inner.read().unwrap().label.clone()
    }

    /// Optional embedding vector for re-identification.
    #[getter]
    fn embedding<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        let det = self.inner.read().unwrap();
        det.embedding
            .as_ref()
            .map(|e| vec_to_numpy1(py, e))
    }

    /// Points in absolute coordinates (world coordinates after camera motion compensation).
    #[getter]
    fn absolute_points<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let det = self.inner.read().unwrap();
        let abs_pts = det.get_absolute_points();
        Ok(dmatrix_to_numpy(py, abs_pts))
    }

    /// Number of points in this detection.
    fn num_points(&self) -> usize {
        self.inner.read().unwrap().num_points()
    }

    /// Dimensionality of points (typically 2 for 2D tracking).
    fn num_dims(&self) -> usize {
        self.inner.read().unwrap().num_dims()
    }

    fn __repr__(&self) -> String {
        let det = self.inner.read().unwrap();
        format!(
            "Detection(points=({}, {}), label={:?})",
            det.num_points(),
            det.num_dims(),
            det.label
        )
    }

    /// Convert to a native norfair.Detection if norfair is installed.
    ///
    /// Returns:
    ///     A norfair.Detection object with the same data.
    ///
    /// Raises:
    ///     ImportError: If norfair is not installed.
    fn to_norfair(&self, py: Python<'_>) -> PyResult<PyObject> {
        let norfair = py.import_bound("norfair")?;
        let detection_cls = norfair.getattr("Detection")?;

        // Get our data
        let points = self.points(py)?;
        let scores = self.scores(py);
        let label = self.label();
        let embedding = self.embedding(py);

        // Build kwargs
        let kwargs = pyo3::types::PyDict::new_bound(py);
        kwargs.set_item("points", points)?;
        if let Some(s) = scores {
            kwargs.set_item("scores", s)?;
        }
        if let Some(l) = label {
            kwargs.set_item("label", l)?;
        }
        if let Some(e) = embedding {
            kwargs.set_item("embedding", e)?;
        }

        // Create norfair Detection
        detection_cls.call((), Some(&kwargs)).map(|obj| obj.into())
    }
}

/// Helper to convert a numpy array to DMatrix
pub fn numpy_to_dmatrix(arr: &PyReadonlyArray2<f64>) -> DMatrix<f64> {
    let arr = arr.as_array();
    let n_rows = arr.nrows();
    let n_cols = arr.ncols();

    let mut data = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            data.push(arr[[i, j]]);
        }
    }

    DMatrix::from_row_slice(n_rows, n_cols, &data)
}

/// Helper to convert DMatrix to numpy array
pub fn dmatrix_to_numpy<'py>(py: Python<'py>, matrix: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
    let (n_rows, n_cols) = (matrix.nrows(), matrix.ncols());

    // Create ndarray Array2 and convert to numpy
    let mut arr = Array2::zeros((n_rows, n_cols));
    for i in 0..n_rows {
        for j in 0..n_cols {
            arr[[i, j]] = matrix[(i, j)];
        }
    }

    arr.into_pyarray_bound(py)
}

/// Helper to convert Vec<f64> to 1D numpy array
pub fn vec_to_numpy1<'py>(py: Python<'py>, data: &[f64]) -> Bound<'py, PyArray1<f64>> {
    let arr = Array1::from_vec(data.to_vec());
    arr.into_pyarray_bound(py)
}
