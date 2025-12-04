//! MOTChallenge metrics evaluation module.
//!
//! This module provides tools for evaluating multi-object tracking performance
//! using the MOTChallenge benchmark format. It includes:
//!
//! - `InformationFile` - Parse seqinfo.ini metadata files
//! - `PredictionsTextFile` - Write tracking results in MOT format
//! - `DetectionFileParser` - Parse detection files
//! - MOT metrics computation (MOTA, MOTP, IDF1, etc.)

mod information_file;
mod predictions;
mod detection_parser;
mod accumulator;
mod evaluation;

pub use information_file::InformationFile;
pub use predictions::PredictionsTextFile;
pub use detection_parser::DetectionFileParser;
pub use accumulator::MOTAccumulator;
pub use evaluation::{eval_mot_challenge, MOTMetrics};
