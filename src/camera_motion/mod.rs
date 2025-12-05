//! Camera motion compensation module.
//!
//! This module provides coordinate transformations for compensating camera motion
//! during object tracking. Supports:
//!
//! - Translation transformations (camera pan/tilt)
//! - Homography transformations (full perspective, requires OpenCV)
//! - Motion estimation from optical flow (requires OpenCV)

mod transformations;

pub use transformations::{
    CoordinateTransformation, NilCoordinateTransformation, TransformationGetter,
    TranslationTransformation, TranslationTransformationGetter,
};

#[cfg(feature = "opencv")]
pub use transformations::{HomographyTransformation, HomographyTransformationGetter};

#[cfg(feature = "opencv")]
mod estimator;

#[cfg(feature = "opencv")]
pub use estimator::MotionEstimator;
