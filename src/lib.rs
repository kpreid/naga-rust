/*!
Backend for generating Rust code.

This backend has limited functionality and is not intended to run arbitrary
shaders. Instead, it is intended to allow relatively simple mathematical functions
to be written once and then executed on both the CPU, via translation to Rust, and GPU,
via translation to a normal shader language, when they are needed in both places.
*/

#![no_std]

extern crate alloc;

//use alloc::format;
use alloc::string::String;

mod conv;
mod types;
mod writer;

use core::fmt;

/// The version of Naga we are compatible with.
pub use naga;

pub use writer::{Writer, WriterFlags};

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

struct Baked(naga::Handle<naga::Expression>);

impl core::fmt::Display for Baked {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.write_prefixed(f, "_e")
    }
}

trait LevelNext {
    fn next(self) -> Self;
}
impl LevelNext for naga::back::Level {
    fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}
