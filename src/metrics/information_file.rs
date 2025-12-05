//! MOTChallenge seqinfo.ini parser.

use crate::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parser for MOTChallenge seqinfo.ini files.
///
/// These files contain metadata about video sequences in the format:
/// ```ini
/// [Sequence]
/// name=MOT17-02-FRCNN
/// imDir=img1
/// frameRate=30
/// seqLength=600
/// imWidth=1920
/// imHeight=1080
/// imExt=.jpg
/// ```
#[derive(Debug)]
pub struct InformationFile {
    path: String,
    lines: Vec<String>,
}

impl InformationFile {
    /// Create a new InformationFile by reading the given file path.
    pub fn new<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let path = file_path.as_ref().to_string_lossy().to_string();
        let file = File::open(&file_path).map_err(|e| {
            Error::IoError(std::io::Error::new(
                e.kind(),
                format!("failed to open information file '{}': {}", path, e),
            ))
        })?;

        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map_while(|l| l.ok()).collect();

        Ok(Self { path, lines })
    }

    /// Search for a variable in the information file.
    ///
    /// # Arguments
    /// * `variable_name` - The key to search for (e.g., "seqLength", "frameRate")
    ///
    /// # Returns
    /// The value as a string, or an error if not found.
    pub fn search(&self, variable_name: &str) -> Result<String> {
        for line in &self.lines {
            if line.starts_with(variable_name) {
                if let Some(equal_idx) = line.find('=') {
                    let value = line[equal_idx + 1..].trim().to_string();
                    return Ok(value);
                }
            }
        }

        Err(Error::MetricsError(format!(
            "couldn't find '{}' in {}",
            variable_name, self.path
        )))
    }

    /// Search for a variable and parse it as an integer.
    pub fn search_int(&self, variable_name: &str) -> Result<i32> {
        let value = self.search(variable_name)?;
        value.parse().map_err(|e| {
            Error::MetricsError(format!(
                "value for '{}' is not an integer: {}",
                variable_name, e
            ))
        })
    }

    /// Search for a variable and return it as a string.
    pub fn search_string(&self, variable_name: &str) -> Result<String> {
        self.search(variable_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_temp_seqinfo() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "[Sequence]").unwrap();
        writeln!(file, "name=MOT17-02-FRCNN").unwrap();
        writeln!(file, "imDir=img1").unwrap();
        writeln!(file, "frameRate=30").unwrap();
        writeln!(file, "seqLength=600").unwrap();
        writeln!(file, "imWidth=1920").unwrap();
        writeln!(file, "imHeight=1080").unwrap();
        file
    }

    #[test]
    fn test_search_int() {
        let file = create_temp_seqinfo();
        let info = InformationFile::new(file.path()).unwrap();

        assert_eq!(info.search_int("seqLength").unwrap(), 600);
        assert_eq!(info.search_int("frameRate").unwrap(), 30);
    }

    #[test]
    fn test_search_string() {
        let file = create_temp_seqinfo();
        let info = InformationFile::new(file.path()).unwrap();

        assert_eq!(info.search_string("name").unwrap(), "MOT17-02-FRCNN");
        assert_eq!(info.search_string("imDir").unwrap(), "img1");
    }

    #[test]
    fn test_search_not_found() {
        let file = create_temp_seqinfo();
        let info = InformationFile::new(file.path()).unwrap();

        assert!(info.search("nonexistent").is_err());
    }
}
