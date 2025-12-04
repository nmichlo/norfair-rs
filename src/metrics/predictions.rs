//! MOTChallenge predictions file writer.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use crate::{Error, Result, TrackedObject};
use super::InformationFile;

/// Writer for tracking predictions in MOTChallenge format.
///
/// The output format is CSV with columns:
/// `frame,id,bb_left,bb_top,bb_width,bb_height,-1,-1,-1,-1`
pub struct PredictionsTextFile {
    length: i32,
    writer: BufWriter<File>,
    frame_number: i32,
}

impl PredictionsTextFile {
    /// Create a new predictions file writer.
    ///
    /// # Arguments
    /// * `input_path` - Path to the sequence being processed
    /// * `save_path` - Directory where predictions/ folder will be created
    /// * `information_file` - Optional InformationFile (if None, loads from input_path/seqinfo.ini)
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        input_path: P1,
        save_path: P2,
        information_file: Option<&InformationFile>,
    ) -> Result<Self> {
        let input_path = input_path.as_ref();
        let save_path = save_path.as_ref();

        // Extract sequence name from input path
        let file_name = input_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("output");

        // Load information file if not provided
        let length = if let Some(info) = information_file {
            info.search_int("seqLength")?
        } else {
            let seqinfo_path = input_path.join("seqinfo.ini");
            let info = InformationFile::new(&seqinfo_path)?;
            info.search_int("seqLength")?
        };

        // Create predictions folder
        let predictions_folder = save_path.join("predictions");
        fs::create_dir_all(&predictions_folder).map_err(|e| {
            Error::IoError(std::io::Error::new(
                e.kind(),
                format!("failed to create predictions folder: {}", e),
            ))
        })?;

        // Open output file
        let out_file_name = predictions_folder.join(format!("{}.txt", file_name));
        let file = File::create(&out_file_name).map_err(|e| {
            Error::IoError(std::io::Error::new(
                e.kind(),
                format!("failed to create output file: {}", e),
            ))
        })?;

        Ok(Self {
            length,
            writer: BufWriter::new(file),
            frame_number: 1,
        })
    }

    /// Get the sequence length.
    pub fn length(&self) -> i32 {
        self.length
    }

    /// Write tracked object information for the current frame.
    ///
    /// # Arguments
    /// * `predictions` - List of TrackedObject instances
    /// * `frame_number` - Optional frame number (if None, uses auto-incremented counter)
    ///
    /// Format: `frame_number,id,bb_left,bb_top,bb_width,bb_height,-1,-1,-1,-1`
    pub fn update(
        &mut self,
        predictions: &[&TrackedObject],
        frame_number: Option<i32>,
    ) -> Result<()> {
        let frame = frame_number.unwrap_or(self.frame_number);

        for obj in predictions {
            // Get bounding box from estimate (expects 2 points: top-left, bottom-right)
            let estimate = &obj.estimate;
            if estimate.nrows() >= 2 && estimate.ncols() >= 2 {
                let x1 = estimate[(0, 0)];
                let y1 = estimate[(0, 1)];
                let x2 = estimate[(1, 0)];
                let y2 = estimate[(1, 1)];

                let bb_left = x1.min(x2);
                let bb_top = y1.min(y2);
                let bb_width = (x2 - x1).abs();
                let bb_height = (y2 - y1).abs();

                // Get object ID (use global_id if id is None)
                let id = obj.id.unwrap_or(obj.global_id);

                writeln!(
                    self.writer,
                    "{},{},{:.2},{:.2},{:.2},{:.2},-1,-1,-1,-1",
                    frame, id, bb_left, bb_top, bb_width, bb_height
                )
                .map_err(|e| Error::IoError(e))?;
            }
        }

        self.frame_number += 1;
        Ok(())
    }

    /// Flush the writer.
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().map_err(|e| Error::IoError(e))
    }
}

impl Drop for PredictionsTextFile {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
