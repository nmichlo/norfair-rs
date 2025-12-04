//! # Norfair - Object Tracking Library
//!
//! Rust port of the Python [norfair](https://github.com/tryolabs/norfair) library.
//!
//! Norfair is a customizable lightweight library for real-time multi-object tracking.
//!
//! ## Features
//!
//! - Kalman filter-based tracking with multiple filter implementations
//! - Pluggable distance functions (Euclidean, IoU, custom)
//! - Camera motion compensation
//! - Re-identification (ReID) support
//! - MOTChallenge metrics evaluation
//!
//! ## Example
//!
//! ```rust,ignore
//! use norfair_rs::{Tracker, TrackerConfig, Detection, distance_by_name};
//!
//! // Create tracker
//! let config = TrackerConfig::new(distance_by_name("euclidean"), 50.0);
//! let mut tracker = Tracker::new(config).unwrap();
//!
//! // Process detections
//! let detections = vec![Detection::new(vec![[100.0, 100.0]])];
//! let tracked_objects = tracker.update(detections, 1, None);
//! ```

// Internal modules (ports of scipy, filterpy, numpy, motmetrics)
pub(crate) mod internal;

// Public modules
pub mod filter;
pub mod distances;
pub mod tracker;
pub mod detection;
pub mod tracked_object;
pub mod matching;
pub mod camera_motion;
pub mod metrics;
pub mod utils;

// Optional modules
#[cfg(feature = "opencv")]
pub mod video;

#[cfg(feature = "opencv")]
pub mod drawing;

// Re-exports for convenience
pub use detection::Detection;
pub use tracked_object::{TrackedObject, TrackedObjectFactory};
pub use tracker::{Tracker, TrackerConfig};
pub use filter::{Filter, FilterFactory};
pub use distances::{Distance, distance_by_name};
pub use camera_motion::CoordinateTransformation;

// Error types
pub use crate::error::{Error, Result};

mod error {
    use thiserror::Error;

    /// Errors that can occur in the norfair library
    #[derive(Error, Debug)]
    pub enum Error {
        #[error("Invalid configuration: {0}")]
        InvalidConfig(String),

        #[error("Invalid detection: {0}")]
        InvalidDetection(String),

        #[error("Invalid points shape: expected {expected}, got {got}")]
        InvalidPointsShape { expected: String, got: String },

        #[error("Distance function error: {0}")]
        DistanceError(String),

        #[error("Filter error: {0}")]
        FilterError(String),

        #[error("Unknown distance function: {0}")]
        UnknownDistance(String),

        #[error("Coordinate transformation error: {0}")]
        TransformError(String),

        #[error("Metrics evaluation error: {0}")]
        MetricsError(String),

        #[error("IO error: {0}")]
        IoError(#[from] std::io::Error),
    }

    /// Result type for norfair operations
    pub type Result<T> = std::result::Result<T, Error>;
}
