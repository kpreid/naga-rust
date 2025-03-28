//! [`naga`] backend allowing you to translate shader code in any language supported by Naga
//! to Rust code.
//!
//! The generated code requires the [`naga_rust_rt`] library.
//! Alternatively, you can use [`naga_rust_embed`], which combines this library with
//! [`naga_rust_rt`] and provides convenient macros for embedding translated WGSL in your Rust code.
//!
//! This library is in an early stage of development and many features do not work yet;
//! this may be indicated by returned errors or by the generated code failing to compile.
//! Broadly, simple mathematical functions will work, and bindings, textures, atomics,
//! derivatives, and workgroup operations will not.
//!
//! [`naga_rust_rt`]: https://docs.rs/naga-rust-rt
//! [`naga_rust_embed`]: https://docs.rs/naga-rust-embed

#![no_std]

extern crate alloc;

use alloc::string::String;
use core::fmt;

use naga::valid::Capabilities;

// -------------------------------------------------------------------------------------------------

mod config;
mod conv;
mod util;
mod writer;

pub use config::Config;
pub use writer::Writer;

/// The version of Naga we are compatible with.
pub use naga;

// -------------------------------------------------------------------------------------------------

/// The [`Capabilities`] supported by our Rust runtime library.
///
/// Pass this to [`naga::valid::Validator`] when validating a module that is to be translated to
/// Rust.
// TODO: There are probably some additional capabilities which should be enabled here
// either because we can support them or they don’t affect us.
pub const CAPABILITIES: Capabilities = Capabilities::FLOAT64;

/// Errors returned by the Rust-generating backend.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The provided [`fmt::Write`] implementation returned an error.
    FmtError(fmt::Error),

    /// The Rust backend currently does not support this particular shader functionality.
    // TODO: this should not be a thing when finished; everything should be either supported
    // or fall into a well-defined category of unsupportedness.
    Unimplemented(String),

    /// We don’t (yet) support texture operations in Rust,
    /// and this is a notably broad category so it gets its own variant.
    #[non_exhaustive]
    TexturesAreUnsupported {
        /// The specific kind of thing found that is unsupported.
        /// Represented as a WGSL-flavored string.
        found: &'static str,
    },

    /// To use a shader with global variables, [`Config::global_struct()`] must be set.
    #[non_exhaustive]
    GlobalVariablesNotEnabled {
        /// The name of one of the prohibited global variables.
        example: String,
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
            Error::Unimplemented(msg) => write!(f, "not yet implemented for Rust: {msg}"),
            Error::TexturesAreUnsupported { found } => {
                write!(f, "texture operations, such as {found}, are not supported")
            }
            Error::GlobalVariablesNotEnabled { example } => write!(
                f,
                "global variable `{example}` found in shader, but not enabled in Config"
            ),
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
    config: Config,
) -> Result<String, Error> {
    let mut w = Writer::new(config);
    let mut output = String::new();
    w.write(&mut output, module, info)?;
    Ok(output)
}
