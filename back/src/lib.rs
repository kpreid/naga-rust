//! [`naga`] backend allowing you to translate shader (GPU) code in any language supported by Naga
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
//! * Textures are supported but texture filtering (use of samplers) is not.
//! * Storage buffers are not supported.
//! * Atomics, derivatives, and workgroup operations are not supported.
//! * Not only are whole pipelines not supported, there is no implementation of interpolation
//!   (as would occur when passing data from a vertex shader to a fragment shader).
//!
//! [`naga_rust_rt`]: https://docs.rs/naga-rust-rt
//! [`naga_rust_embed`]: https://docs.rs/naga-rust-embed

#![no_std]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use naga::valid::Capabilities;

use crate::ra::PrintAst as _;

// -------------------------------------------------------------------------------------------------

mod config;
mod conv;
mod logic;
mod ra;
mod translator;
mod util;

pub use config::{Condition, Config, Effect, Inline, Rule};
pub use translator::Translator;

/// The version of Naga we are compatible with.
pub use naga;

// -------------------------------------------------------------------------------------------------

/// A shader program ([`naga::Module`]) translated to Rust.
///
/// Obtain this by calling [`Translator::translate_module()`].
///
/// Produces a string of Rust code when formatted with [`fmt::Display`] or [`to_string()`].
/// The code consists solely of [items], and can be placed inside a module or block.
///
/// [`to_string()`]: alloc::string::ToString::to_string
/// [items]: https://doc.rust-lang.org/reference/items.html
pub struct Rust {
    runtime_path: Cow<'static, str>,
    top_level_items: Vec<ra::Item>,
}

impl fmt::Display for Rust {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let print_ctx = ra::PrintCtx {
            runtime_path: &self.runtime_path,
            indent: naga::back::Level(0),
        };
        for item in &self.top_level_items {
            item.write(f, print_ctx)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Rust {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Rust(")?;
        let print_ctx = ra::PrintCtx {
            runtime_path: &self.runtime_path,
            indent: naga::back::Level(1),
        };
        for item in &self.top_level_items {
            item.write(f, print_ctx)?;
        }
        // Items format with trailing newlines, so we don’t add one here.
        write!(f, ")")
    }
}

// -------------------------------------------------------------------------------------------------

/// The [`Capabilities`] supported by our Rust runtime library.
///
/// Pass this to [`naga::valid::Validator`] when validating a module that is to be translated to
/// Rust.
// TODO: There are probably some additional capabilities which should be enabled here
// either because we can support them or they don’t affect us.
pub const CAPABILITIES: Capabilities = Capabilities::FLOAT64;

/// Errors indicating that a shader cannot be translated to Rust by [`Translator`].
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
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

impl core::error::Error for Error {}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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

// -------------------------------------------------------------------------------------------------

/// Converts `module` to a string of Rust code.
///
/// This is a convenience wrapper around [`Translator::translate_module()`].
///
/// # Errors
///
/// Returns an error if the module cannot be represented as Rust.
pub fn write_string(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    config: Config,
) -> Result<String, Error> {
    Ok(format!(
        "{}",
        Translator::new(config).translate_module(module, info)?
    ))
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
    let mut w = Translator::new(config.clone());
    let program = w.translate_module(module, info)?;

    let functions: Vec<ra::FunctionItem> = program
        .top_level_items
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
    function
        .body
        .write(
            &mut output,
            ra::PrintCtx {
                runtime_path: &config.runtime_path,
                indent: naga::back::Level(0),
            },
        )
        .unwrap();
    Ok(output)
}
