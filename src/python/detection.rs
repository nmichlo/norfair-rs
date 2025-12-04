//! Python wrapper for Detection.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyArrayMethods, IntoPyArray, PyUntypedArrayMethods};
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
    ///     points: Detection points as a numpy array of shape (n_points, n_dims) or (n_dims,).
    ///             For keypoints: [[x1, y1], [x2, y2], ...]
    ///             For bounding boxes: [[x1, y1], [x2, y2]] (top-left, bottom-right)
    ///             1D arrays like [x, y] are automatically reshaped to [[x, y]].
    ///             Arrays will be converted to float64 dtype.
    ///     scores: Optional per-point confidence scores of shape (n_points,).
    ///     data: Optional arbitrary user data (not currently used in norfair_rs).
    ///     label: Optional class label for multi-class tracking.
    ///     embedding: Optional embedding vector for re-identification.
    #[new]
    #[pyo3(signature = (points, scores=None, data=None, label=None, embedding=None))]
    #[allow(unused_variables)]
    fn new(
        py: Python<'_>,
        points: &Bound<'_, pyo3::types::PyAny>,
        scores: Option<&Bound<'_, pyo3::types::PyAny>>,
        data: Option<PyObject>,
        label: Option<String>,
        embedding: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<Self> {
        // Convert points to float64 numpy array
        let np = py.import_bound("numpy")?;
        let points_f64 = np.call_method1("asarray", (points,))?
            .call_method1("astype", (np.getattr("float64")?,))?;

        // Get ndim to check if we need to reshape
        let ndim: usize = points_f64.getattr("ndim")?.extract()?;
        let points_2d = if ndim == 1 {
            // Reshape 1D array [x, y] to 2D [[x, y]]
            points_f64.call_method1("reshape", ((1, -1),))?
        } else if ndim == 2 {
            points_f64
        } else {
            return Err(PyValueError::new_err(format!(
                "Points must be 1D or 2D array, got {}D",
                ndim
            )));
        };

        // Extract as PyArray2<f64> using bound API
        let points_arr: Bound<'_, PyArray2<f64>> = points_2d.extract()?;
        let points_readonly = points_arr.readonly();
        let points_view = points_readonly.as_array();
        let n_points = points_view.nrows();
        let n_dims = points_view.ncols();

        // Validate dimensions
        if n_dims != 2 && n_dims != 3 {
            return Err(PyValueError::new_err(format!(
                "Points must have 2 or 3 coordinate dimensions, got {}",
                n_dims
            )));
        }

        // Convert to row-major Vec for DMatrix
        let mut data_vec = Vec::with_capacity(n_points * n_dims);
        for i in 0..n_points {
            for j in 0..n_dims {
                data_vec.push(points_view[[i, j]]);
            }
        }
        let points_matrix = DMatrix::from_row_slice(n_points, n_dims, &data_vec);

        // Convert scores if provided
        let scores_vec: Option<Vec<f64>> = if let Some(s) = scores {
            let scores_f64 = np.call_method1("asarray", (s,))?
                .call_method1("astype", (np.getattr("float64")?,))?
                .call_method0("ravel")?;  // Flatten to 1D
            let scores_arr: Bound<'_, PyArray1<f64>> = scores_f64.extract()?;
            let scores_readonly = scores_arr.readonly();
            let scores_view = scores_readonly.as_array();
            Some(scores_view.iter().cloned().collect::<Vec<f64>>())
        } else {
            None
        };

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
        let embedding_vec: Option<Vec<f64>> = if let Some(e) = embedding {
            let emb_f64 = np.call_method1("asarray", (e,))?
                .call_method1("astype", (np.getattr("float64")?,))?
                .call_method0("ravel")?;  // Flatten to 1D
            let emb_arr: Bound<'_, PyArray1<f64>> = emb_f64.extract()?;
            let emb_readonly = emb_arr.readonly();
            let emb_view = emb_readonly.as_array();
            Some(emb_view.iter().cloned().collect::<Vec<f64>>())
        } else {
            None
        };

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
pub fn numpy_to_dmatrix(py: Python<'_>, arr: &Bound<'_, pyo3::types::PyAny>) -> PyResult<DMatrix<f64>> {
    let np = py.import_bound("numpy")?;
    let arr_f64 = np.call_method1("asarray", (arr,))?
        .call_method1("astype", (np.getattr("float64")?,))?;
    let points_arr: Bound<'_, PyArray2<f64>> = arr_f64.extract()?;
    let arr_readonly = points_arr.readonly();
    let arr_view = arr_readonly.as_array();
    let n_rows = arr_view.nrows();
    let n_cols = arr_view.ncols();

    let mut data = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            data.push(arr_view[[i, j]]);
        }
    }

    Ok(DMatrix::from_row_slice(n_rows, n_cols, &data))
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
