//! Detection file parser for MOTChallenge format.

use crate::{Detection, Error, Result};
use nalgebra::DMatrix;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parser for MOTChallenge detection files.
///
/// Reads detection files in CSV format:
/// `frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z`
pub struct DetectionFileParser {
    detections: Vec<Vec<Detection>>,
    current_frame: usize,
}

impl DetectionFileParser {
    /// Create a new detection file parser.
    ///
    /// # Arguments
    /// * `file_path` - Path to the detection file
    /// * `num_frames` - Total number of frames in the sequence
    pub fn new<P: AsRef<Path>>(file_path: P, num_frames: usize) -> Result<Self> {
        let file = File::open(&file_path).map_err(|e| {
            Error::IoError(std::io::Error::new(
                e.kind(),
                format!("failed to open detection file: {}", e),
            ))
        })?;

        let reader = BufReader::new(file);
        let mut detections: Vec<Vec<Detection>> = vec![Vec::new(); num_frames];

        for line_result in reader.lines() {
            let line = line_result.map_err(Error::IoError)?;
            let parts: Vec<&str> = line.split(',').collect();

            if parts.len() < 7 {
                continue; // Skip malformed lines
            }

            // Parse frame number (1-indexed in MOT format)
            let frame: usize = parts[0].trim().parse().unwrap_or(0);
            if frame == 0 || frame > num_frames {
                continue;
            }

            // Parse bounding box
            let bb_left: f64 = parts[2].trim().parse().unwrap_or(0.0);
            let bb_top: f64 = parts[3].trim().parse().unwrap_or(0.0);
            let bb_width: f64 = parts[4].trim().parse().unwrap_or(0.0);
            let bb_height: f64 = parts[5].trim().parse().unwrap_or(0.0);

            // Parse confidence (optional, default to 1.0)
            let conf: f64 = parts
                .get(6)
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1.0);

            // Convert to points (top-left, bottom-right)
            let x1 = bb_left;
            let y1 = bb_top;
            let x2 = bb_left + bb_width;
            let y2 = bb_top + bb_height;

            let points = DMatrix::from_row_slice(2, 2, &[x1, y1, x2, y2]);
            let detection = Detection::with_config(points, Some(vec![conf, conf]), None, None)
                .unwrap_or_default();

            detections[frame - 1].push(detection);
        }

        Ok(Self {
            detections,
            current_frame: 0,
        })
    }

    /// Get detections for a specific frame (0-indexed).
    pub fn get_detections(&self, frame: usize) -> Option<&[Detection]> {
        self.detections.get(frame).map(|v| v.as_slice())
    }

    /// Get the number of frames.
    pub fn num_frames(&self) -> usize {
        self.detections.len()
    }
}

impl Iterator for DetectionFileParser {
    type Item = Vec<Detection>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_frame >= self.detections.len() {
            return None;
        }

        let frame_detections = self.detections[self.current_frame].clone();
        self.current_frame += 1;
        Some(frame_detections)
    }
}
