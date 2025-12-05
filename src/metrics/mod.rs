//! MOTChallenge metrics evaluation module.
//!
//! This module provides tools for evaluating multi-object tracking performance
//! using the MOTChallenge benchmark format. It includes:
//!
//! - `InformationFile` - Parse seqinfo.ini metadata files
//! - `PredictionsTextFile` - Write tracking results in MOT format
//! - `DetectionFileParser` - Parse detection files
//! - MOT metrics computation (MOTA, MOTP, IDF1, etc.)

mod accumulator;
mod detection_parser;
mod evaluation;
mod information_file;
mod predictions;

pub use accumulator::MOTAccumulator;
pub use detection_parser::DetectionFileParser;
pub use evaluation::{eval_mot_challenge, MOTMetrics};
pub use information_file::InformationFile;
pub use predictions::PredictionsTextFile;
