//! [`naga`] backend allowing you to translate shader code in any language supported by Naga
//! to Rust code.
//!
//! The generated code requires the [`naga_rust_rt`] library.
//! Alternatively, you can use [`naga_rust_embed`], which combines this library with
//! [`naga_rust_rt`] and provides convenient macros for embedding translated WGSL in your Rust code.
//!
//! This library is in an early stage of development and many features do not work yet.
//! Expect compilation failures, incorrect behaviors, and to have to tweak your code to fit,
//! if you wish to use it. Broadly:
//!
//! * Simple mathematical functions will work.
//! * Code involving pointers is likely to fail to compile.
//! * Textures are supported but texture filtering is not.
//! * Atomics, derivatives, and workgroup operations are not supported.
//! * Pipelines involving multiple shaders (e.g. passing data from vertex to fragment)
//!   are not automatically executed but you can build that yourself.
//!
//! [`naga_rust_rt`]: https://docs.rs/naga-rust-rt
//! [`naga_rust_embed`]: https://docs.rs/naga-rust-embed

#![no_std]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use naga::valid::Capabilities;

use crate::ra::PrintAst as _;

// -------------------------------------------------------------------------------------------------

mod config;
mod conv;
mod ra;
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

    /// To use a shader with private global variables, [`Config::global_struct()`] must be set.
    #[non_exhaustive]
    GlobalVariablesNotEnabled {
        /// The name of one of the prohibited global variables.
        example: String,
    },

    /// To use a shader with resources, [`Config::resource_struct()`] must be set.
    #[non_exhaustive]
    ResourcesNotEnabled {
        /// The name of one of the prohibited resources.
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
            Error::GlobalVariablesNotEnabled { example } => write!(
                f,
                "global variable `{example}` found in shader, but `global_struct` is not configured"
            ),
            Error::ResourcesNotEnabled { example } => write!(
                f,
                "resource `{example}` found in shader, but `resource_struct` is not configured"
            ),
        }
    }
}

/// Converts `module` to a string of Rust code.
///
/// This is a convenience wrapper around [`Writer::write()`].
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

/// Converts `module` to Rust code, then throws away everything but the body of the single function.
///
/// This function is used to help test the translation of individual statements and expressions
/// without having to reiterate the rest of the generated code.
#[doc(hidden)] // test helper
#[mutants::skip] // test helper, not code under test
pub fn translate_function_body_only_for_testing(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    config: &Config,
) -> Result<String, Error> {
    let mut w = Writer::new(config.clone());
    let items = w.translate_module(module, info)?;

    let functions: Vec<ra::FunctionItem> = items
        .into_iter()
        .filter_map(|item| {
            let ra::Item::Function(fn_item) = item else {
                return None;
            };
            // Look for the internal/vectorized function rather than the wrapper function.
            if !fn_item.name.starts_with("v_") {
                return None;
            }
            Some(fn_item)
        })
        .collect();

    if functions.len() != 1 {
        return Err(Error::Unimplemented(format!(
            "expected exactly one function; found {items:?}",
            items = functions.into_iter().map(|f| f.name).collect::<Vec<_>>()
        )));
    }
    let function = functions.into_iter().next().unwrap();

    let mut output = String::new();
    function.body.write(
        &mut output,
        ra::PrintCtx {
            config,
            indent: naga::back::Level(0),
        },
    )?;
    Ok(output)
}
