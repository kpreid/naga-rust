//! [`naga`] backend allowing you to translate shader code in any language supported by Naga
//! to Rust code.
//!
//! The generated code requires the `naga-rust` library.
//! You should probably also use that library’s macros to embed code in your project,
//! unless you have special code generation requirements or you want to do it in a build
//! script instead of macros.

#![no_std]

extern crate alloc;

use alloc::string::String;
use core::fmt;

// -------------------------------------------------------------------------------------------------

mod conv;
mod util;
mod writer;

pub use writer::{Writer, WriterFlags};

/// The version of Naga we are compatible with.
pub use naga;

// -------------------------------------------------------------------------------------------------

/// Errors returned by the Rust-generating backend.
#[derive(Debug)]
#[allow(missing_docs, reason = "TODO: clean up and document")]
#[non_exhaustive]
pub enum Error {
    /// The provided [`fmt::Write`] implementation returned an error.
    FmtError(fmt::Error),

    // TODO: this should not be a thing when finished
    //#[error("{0}")]
    Unimplemented(String),

    /// We don’t (yet) support texture operations in Rust,
    /// and this is a notably broad category.
    TexturesAreUnsupported {
        /// The specific kind of thing found that is unsupported.
        /// Represented as a WGSL-flavored string.
        found: &'static str,
    },
}

impl From<fmt::Error> for Error {
    fn from(value: fmt::Error) -> Self {
        Self::FmtError(value)
    }
}

impl core::error::Error for Error {}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::FmtError(fmt::Error) => write!(f, "formatting cancelled"),
            Error::Unimplemented(msg) => write!(f, "{msg}"),
            Error::TexturesAreUnsupported { found } => {
                write!(f, "texture operations, such as {found}, are not supported")
            }
        }
    }
}

/// Converts `module` to a string of Rust code.
///
/// # Errors
///
/// Returns an error if the module cannot be represented as Rust.
pub fn write_string(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    flags: WriterFlags,
) -> Result<String, Error> {
    let mut w = Writer::new(String::new(), flags);
    w.write(module, info)?;
    let output = w.finish();
    Ok(output)
}
