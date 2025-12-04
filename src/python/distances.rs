//! Python wrappers for distance functions.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::PyArray2;
use nalgebra::DMatrix;

use crate::distances::{
    Distance,
    distance_by_name as rust_distance_by_name,
    try_distance_by_name,
    iou as rust_iou,
};
use crate::{Detection, TrackedObject};

use super::detection::{PyDetection, numpy_to_dmatrix, dmatrix_to_numpy};
use super::tracked_object::PyTrackedObject;

/// Enum to hold different distance implementations
pub enum PyDistanceEnum {
    /// Built-in Rust distance function
    Builtin(Box<dyn Distance>),
    /// Python callable for scalar distance
    PyScalar(Py<PyAny>),
    /// Python callable for vectorized distance
    PyVectorized(Py<PyAny>),
}

impl PyDistanceEnum {
    /// Compute distances using this distance function.
    pub fn get_distances(
        &self,
        py: Python<'_>,
        objects: &[&TrackedObject],
        candidates: &[&Detection],
    ) -> PyResult<DMatrix<f64>> {
        match self {
            PyDistanceEnum::Builtin(dist) => {
                Ok(dist.get_distances(objects, candidates))
            }
            PyDistanceEnum::PyScalar(func) => {
                // Call Python function for each pair
                let n_candidates = candidates.len();
                let n_objects = objects.len();
                let mut result = DMatrix::zeros(n_candidates, n_objects);

                for (i, cand) in candidates.iter().enumerate() {
                    for (j, obj) in objects.iter().enumerate() {
                        let py_det = PyDetection::from_detection((*cand).clone());
                        let py_obj = PyTrackedObject::from_tracked_object(obj);

                        let distance: f64 = func
                            .call1(py, (py_det, py_obj))?
                            .extract(py)?;
                        result[(i, j)] = distance;
                    }
                }
                Ok(result)
            }
            PyDistanceEnum::PyVectorized(func) => {
                // Call Python function with batched arrays
                let n_candidates = candidates.len();
                let n_objects = objects.len();

                if n_candidates == 0 || n_objects == 0 {
                    return Ok(DMatrix::zeros(n_candidates, n_objects));
                }

                // Build candidate points array (flatten all points)
                let cand_points: Vec<Vec<f64>> = candidates
                    .iter()
                    .map(|c| {
                        let pts = &c.points;
                        let mut flat = Vec::with_capacity(pts.nrows() * pts.ncols());
                        for i in 0..pts.nrows() {
                            for j in 0..pts.ncols() {
                                flat.push(pts[(i, j)]);
                            }
                        }
                        flat
                    })
                    .collect();

                // Build object estimates array
                let obj_points: Vec<Vec<f64>> = objects
                    .iter()
                    .map(|o| {
                        let pts = &o.estimate;
                        let mut flat = Vec::with_capacity(pts.nrows() * pts.ncols());
                        for i in 0..pts.nrows() {
                            for j in 0..pts.ncols() {
                                flat.push(pts[(i, j)]);
                            }
                        }
                        flat
                    })
                    .collect();

                // Convert to numpy arrays
                let dim = cand_points.first().map(|v| v.len()).unwrap_or(0);
                let cand_arr = PyArray2::from_vec2_bound(py, &cand_points)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let obj_arr = PyArray2::from_vec2_bound(py, &obj_points)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;

                // Call the Python function
                let result_obj = func.call1(py, (cand_arr, obj_arr))?;

                Ok(numpy_to_dmatrix(py, result_obj.bind(py))?)
            }
        }
    }
}

// Can't implement Send+Sync for PyDistanceEnum directly due to Py<PyAny>
// Instead, we use it within GIL-protected contexts

/// Wrapper for scalar distance functions.
///
/// Scalar distance functions compute the distance between a single
/// Detection and TrackedObject pair.
#[pyclass(name = "ScalarDistance")]
pub struct PyScalarDistance {
    pub(crate) func: Py<PyAny>,
}

#[pymethods]
impl PyScalarDistance {
    /// Create a new ScalarDistance from a Python callable.
    ///
    /// Args:
    ///     distance_function: A callable that takes (Detection, TrackedObject)
    ///                        and returns a float distance.
    #[new]
    fn new(distance_function: Py<PyAny>) -> Self {
        Self {
            func: distance_function,
        }
    }

    fn __repr__(&self) -> String {
        "ScalarDistance(<callable>)".to_string()
    }
}

/// Wrapper for vectorized distance functions.
///
/// Vectorized distance functions compute distances between batches
/// of detections and tracked objects efficiently.
#[pyclass(name = "VectorizedDistance")]
pub struct PyVectorizedDistance {
    pub(crate) func: Py<PyAny>,
}

#[pymethods]
impl PyVectorizedDistance {
    /// Create a new VectorizedDistance from a Python callable.
    ///
    /// Args:
    ///     distance_function: A callable that takes (candidates_array, objects_array)
    ///                        and returns a distance matrix of shape (n_candidates, n_objects).
    #[new]
    fn new(distance_function: Py<PyAny>) -> Self {
        Self {
            func: distance_function,
        }
    }

    fn __repr__(&self) -> String {
        "VectorizedDistance(<callable>)".to_string()
    }
}

/// Get a built-in distance function by name.
///
/// Supported names:
///     - "euclidean", "manhattan", "cosine", "chebyshev" - scipy metrics
///     - "frobenius" - Frobenius norm of difference
///     - "mean_euclidean" - Mean L2 distance per point
///     - "mean_manhattan" - Mean L1 distance per point
///     - "iou" - Intersection over Union for bounding boxes
///
/// Args:
///     name: Name of the distance function.
///
/// Returns:
///     A distance function object.
///
/// Raises:
///     ValueError: If the distance name is not recognized.
#[pyfunction]
pub fn get_distance_by_name(name: &str) -> PyResult<PyBuiltinDistance> {
    try_distance_by_name(name)
        .map(|d| PyBuiltinDistance { inner: d, name: name.to_string() })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// A built-in distance function.
///
/// This object wraps a built-in distance function and provides a callable interface.
/// Compatible with norfair API where Distance objects have a .distance_function attribute.
#[pyclass(name = "Distance")]
pub struct PyBuiltinDistance {
    pub(crate) inner: Box<dyn Distance>,
    pub(crate) name: String,
}

#[pymethods]
impl PyBuiltinDistance {
    fn __repr__(&self) -> String {
        format!("Distance('{}')", self.name)
    }

    /// The underlying distance function as a callable.
    ///
    /// For scalar distances (frobenius, mean_euclidean, mean_manhattan):
    ///     Takes (detection, tracked_object) and returns float.
    ///
    /// For vectorized distances (iou):
    ///     Takes (candidates_array, objects_array) and returns distance matrix.
    #[getter]
    fn distance_function(slf: PyRef<'_, Self>) -> PyDistanceFunctionWrapper {
        PyDistanceFunctionWrapper {
            name: slf.name.clone(),
        }
    }
}

/// Wrapper that makes the distance function callable.
/// Used as the return value of Distance.distance_function property.
#[pyclass(name = "_DistanceFunction")]
pub struct PyDistanceFunctionWrapper {
    name: String,
}

#[pymethods]
impl PyDistanceFunctionWrapper {
    /// Call the distance function.
    ///
    /// For scalar distances: call(detection, tracked_object) -> float
    /// For vectorized distances: call(candidates, objects) -> np.ndarray
    #[pyo3(signature = (*args))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, pyo3::types::PyTuple>,
    ) -> PyResult<PyObject> {
        match self.name.as_str() {
            "iou" => {
                // Vectorized: expects (candidates, objects) arrays
                if args.len() != 2 {
                    return Err(PyValueError::new_err(
                        "iou distance function expects 2 arguments: (candidates, objects)"
                    ));
                }
                let candidates = args.get_item(0)?;
                let objects = args.get_item(1)?;
                let result = iou_inner(py, &candidates, &objects)?;
                Ok(result.into())
            }
            _ => {
                // Scalar: expects (detection, tracked_object)
                if args.len() != 2 {
                    return Err(PyValueError::new_err(format!(
                        "{} distance function expects 2 arguments: (detection, tracked_object)",
                        self.name
                    )));
                }
                let detection: PyRef<'_, PyDetection> = args.get_item(0)?.extract()?;
                let tracked_obj: PyRef<'_, PyTrackedObject> = args.get_item(1)?.extract()?;

                let det = detection.get_detection();
                let estimate = &tracked_obj.estimate;

                let distance = match self.name.as_str() {
                    "frobenius" => {
                        let diff = &det.points - estimate;
                        diff.norm()
                    }
                    "mean_euclidean" => {
                        let n_points = det.points.nrows();
                        if n_points == 0 {
                            0.0
                        } else {
                            let mut total = 0.0;
                            for i in 0..n_points {
                                let mut point_dist = 0.0;
                                for j in 0..det.points.ncols() {
                                    let diff = det.points[(i, j)] - estimate[(i, j)];
                                    point_dist += diff * diff;
                                }
                                total += point_dist.sqrt();
                            }
                            total / n_points as f64
                        }
                    }
                    "mean_manhattan" => {
                        let n_points = det.points.nrows();
                        if n_points == 0 {
                            0.0
                        } else {
                            let mut total = 0.0;
                            for i in 0..n_points {
                                for j in 0..det.points.ncols() {
                                    total += (det.points[(i, j)] - estimate[(i, j)]).abs();
                                }
                            }
                            total / n_points as f64
                        }
                    }
                    name => {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported scalar distance: {}. Use get_distance_by_name() for supported distances.",
                            name
                        )));
                    }
                };
                Ok(distance.into_py(py))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("<distance_function '{}'>", self.name)
    }
}

/// Extract a distance function from various Python inputs.
///
/// Supports:
/// - String names (e.g., "iou", "euclidean")
/// - PyBuiltinDistance objects
/// - PyScalarDistance objects (Python callables wrapped)
/// - PyVectorizedDistance objects (Python callables wrapped)
/// - Raw Python callables (auto-wrapped as ScalarDistance)
pub fn extract_distance(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<PyDistanceEnum> {
    // Try string first
    if let Ok(name) = obj.extract::<String>() {
        let dist = try_distance_by_name(&name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok(PyDistanceEnum::Builtin(dist));
    }

    // Try built-in distance
    if let Ok(d) = obj.extract::<PyRef<PyBuiltinDistance>>() {
        // We need to clone the inner distance - use distance_by_name as a workaround
        let dist = rust_distance_by_name(&d.name);
        return Ok(PyDistanceEnum::Builtin(dist));
    }

    // Try scalar distance wrapper
    if let Ok(d) = obj.extract::<PyRef<PyScalarDistance>>() {
        return Ok(PyDistanceEnum::PyScalar(d.func.clone_ref(py)));
    }

    // Try vectorized distance wrapper
    if let Ok(d) = obj.extract::<PyRef<PyVectorizedDistance>>() {
        return Ok(PyDistanceEnum::PyVectorized(d.func.clone_ref(py)));
    }

    // Try as a callable (assume scalar distance)
    if obj.is_callable() {
        return Ok(PyDistanceEnum::PyScalar(obj.clone().unbind()));
    }

    Err(PyValueError::new_err(
        "distance_function must be a string, Distance, ScalarDistance, VectorizedDistance, or callable"
    ))
}

// ===== Built-in distance functions exposed to Python =====

/// Frobenius norm distance between detection and tracked object.
///
/// Computes the Frobenius norm of the difference between detection points
/// and the tracked object's estimate.
#[pyfunction]
pub fn frobenius(detection: &PyDetection, tracked_object: &PyTrackedObject) -> f64 {
    let det = detection.get_detection();

    // Create a minimal TrackedObject-like struct for the distance calculation
    let diff = &det.points - &tracked_object.estimate;
    diff.norm()
}

/// Mean Euclidean distance between detection and tracked object.
///
/// Computes the mean L2 distance across all corresponding points.
#[pyfunction]
pub fn mean_euclidean(detection: &PyDetection, tracked_object: &PyTrackedObject) -> f64 {
    let det = detection.get_detection();
    let n_points = det.points.nrows();

    if n_points == 0 {
        return 0.0;
    }

    let mut total_dist = 0.0;
    for i in 0..n_points {
        let mut point_dist = 0.0;
        for j in 0..det.points.ncols() {
            let diff = det.points[(i, j)] - tracked_object.estimate[(i, j)];
            point_dist += diff * diff;
        }
        total_dist += point_dist.sqrt();
    }

    total_dist / n_points as f64
}

/// Mean Manhattan distance between detection and tracked object.
///
/// Computes the mean L1 distance across all corresponding points.
#[pyfunction]
pub fn mean_manhattan(detection: &PyDetection, tracked_object: &PyTrackedObject) -> f64 {
    let det = detection.get_detection();
    let n_points = det.points.nrows();

    if n_points == 0 {
        return 0.0;
    }

    let mut total_dist = 0.0;
    for i in 0..n_points {
        for j in 0..det.points.ncols() {
            total_dist += (det.points[(i, j)] - tracked_object.estimate[(i, j)]).abs();
        }
    }

    total_dist / n_points as f64
}

/// Inner implementation of IoU that accepts PyAny.
fn iou_inner<'py>(
    py: Python<'py>,
    candidates: &Bound<'py, pyo3::types::PyAny>,
    objects: &Bound<'py, pyo3::types::PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cand_mat = numpy_to_dmatrix(py, candidates)?;
    let obj_mat = numpy_to_dmatrix(py, objects)?;

    // Validate input dimensions - raise AssertionError to match norfair behavior
    if cand_mat.ncols() < 4 {
        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
            "IoU requires at least 4 columns (x1, y1, x2, y2), got {}",
            cand_mat.ncols()
        )));
    }
    if obj_mat.ncols() < 4 {
        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
            "IoU requires at least 4 columns (x1, y1, x2, y2), got {}",
            obj_mat.ncols()
        )));
    }
    // Also check for too many columns (norfair uses exactly 4)
    if cand_mat.ncols() != 4 {
        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
            "IoU expects exactly 4 columns (x1, y1, x2, y2), got {}",
            cand_mat.ncols()
        )));
    }

    let result = rust_iou(&cand_mat, &obj_mat);
    Ok(dmatrix_to_numpy(py, &result))
}

/// Intersection over Union (IoU) distance for bounding boxes.
///
/// Args:
///     candidates: Array of candidate bounding boxes, shape (n_candidates, 4).
///                 Each row is [x1, y1, x2, y2].
///     objects: Array of object bounding boxes, shape (n_objects, 4).
///              Each row is [x1, y1, x2, y2].
///
/// Returns:
///     Distance matrix of shape (n_candidates, n_objects).
///     Distance is 1 - IoU, so 0 means perfect overlap.
#[pyfunction]
pub fn iou<'py>(
    py: Python<'py>,
    candidates: &Bound<'py, pyo3::types::PyAny>,
    objects: &Bound<'py, pyo3::types::PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    iou_inner(py, candidates, objects)
}
