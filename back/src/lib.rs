//! [`naga`] backend allowing you to translate shader code in any language supported by Naga
//! to Rust code.
//!
//! This does not necessarily mean you can run your compute or render pipelines in Rust
//! on your CPU unchanged; this is *not* a “software renderer”. Rather, the primary goal
//! of the project is to allow you to share selected *functions* between CPU and GPU, so
//! that they can agree on definitions that might be executed in either place.

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

#[derive(Debug)]
pub enum Error {
    //#[error(transparent)]
    FmtError(core::fmt::Error),
    //#[error("{0}")]
    Custom(String),
    //#[error("{0}")]
    Unimplemented(String), // TODO: Error used only during development
    //#[error("Unsupported relational function: {0:?}")]
    UnsupportedRelationalFunction(naga::RelationalFunction),
    //#[error("Unsupported {kind}: {value}")]
    Unsupported {
        /// What kind of unsupported thing this is: interpolation, builtin, etc.
        kind: &'static str,

        /// The debug form of the Naga IR value that this backend can't express.
        value: String,
    },
}

impl From<fmt::Error> for Error {
    fn from(value: fmt::Error) -> Self {
        Self::FmtError(value)
    }
}

impl core::error::Error for Error {}
impl fmt::Display for Error {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!("bring back error formatting")
    }
}

// TODO: are we going to need this?
// impl Error {
//     /// Produce an [`Unsupported`] error for `value`.
//     ///
//     /// [`Unsupported`]: Error::Unsupported
//     fn unsupported<T: core::fmt::Debug>(kind: &'static str, value: T) -> Error {
//         Error::Unsupported {
//             kind,
//             value: format!("{value:?}"),
//         }
//     }
// }

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
