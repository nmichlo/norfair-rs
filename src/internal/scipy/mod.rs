//! SciPy functions port.
//!
//! Ported from:
//! - scipy.spatial.distance
//! - scipy.optimize
//!
//! License: BSD 3-Clause (see LICENSE file in this directory)

mod distance;
mod optimize;

pub use distance::*;
pub use optimize::*;
