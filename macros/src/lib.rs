//! This is a proc-macro helper library. Don't use this library directly; use [`naga_rust_embed`]
//! instead.
//!
//! [`naga_rust_embed`]: https://docs.rs/naga-rust-embed

#![allow(missing_docs, reason = "not intended to be used directly")]

use std::error::Error;
use std::fmt;

use proc_macro2::Span;
use proc_macro2::TokenStream;

use naga_rust_back::Config;
use naga_rust_back::naga;

// -------------------------------------------------------------------------------------------------

mod parse_config;
use parse_config::ConfigAndStr;

mod macro_util;
use macro_util::MacroError;

#[cfg(test)]
mod tests;

// -------------------------------------------------------------------------------------------------

#[proc_macro]
pub fn include_wgsl_mr(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match ConfigAndStr::parse(input.into()) {
        Ok(ConfigAndStr {
            config,
            string_span: path_span,
            string: path_literal,
        }) => match include_wgsl_mr_impl(config, path_span, &path_literal) {
            Ok(expansion) => expansion.into(),
            Err(error) => error.to_compile_error(),
        },
        Err(e) => e.to_compile_error(),
    }
}

#[proc_macro]
pub fn wgsl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    match ConfigAndStr::parse(input.into()) {
        Ok(ConfigAndStr {
            config,
            string_span: source_span,
            string: source_literal,
        }) => match parse_and_translate(config, source_span, &source_literal) {
            Ok(expansion) => expansion.into(),
            Err(error) => error.to_compile_error(),
        },
        Err(e) => e.to_compile_error(),
    }
}

// -------------------------------------------------------------------------------------------------

/// Implementation of the [`include_wgsl_mr!`] macro.
fn include_wgsl_mr_impl(
    config: Config,
    path_span: Span,
    path_text: &str,
) -> Result<TokenStream, MacroError> {
    let (wgsl_source_text, include_str_call) =
        macro_util::read_file_manifest_relative(path_span, path_text)?;

    let translated_tokens = parse_and_translate(config, path_span, &wgsl_source_text)?;

    Ok(TokenStream::from_iter(
        include_str_call.into_iter().chain(translated_tokens),
    ))
}

/// Parse WGSL text and translate it to Rust.
/// This is directly the implementation of the [`wgsl!`] macro, and used indirectly by
/// [`include_wgsl_mr!`].
fn parse_and_translate(
    config: Config,
    wgsl_source_span: Span,
    wgsl_source_text: &str,
) -> Result<TokenStream, MacroError> {
    let options = naga::front::wgsl::Options {
        parse_doc_comments: config.get_include_documentation(),
        capabilities: naga_rust_back::CAPABILITIES,
    };

    let module: naga::Module = naga::front::wgsl::Frontend::new_with_options(options)
        .parse(wgsl_source_text)
        .map_err(|error| {
            MacroError::new(
                wgsl_source_span,
                format!("failed to parse WGSL text: {}", ErrorChain(&error)),
            )
        })?;

    // TODO: allow the user of the macro to configure which validation is done.
    let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga_rust_back::CAPABILITIES,
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    // TODO: Add support for subgroup operations, then update this.
    .subgroup_operations(naga::valid::SubgroupOperationSet::empty())
    .validate(&module)
    .map_err(|error| {
        MacroError::new(
            wgsl_source_span,
            format!("failed to validate WGSL: {}", ErrorChain(&error)),
        )
    })?;

    let translated_source: String = naga_rust_back::write_string(&module, &module_info, config)
        .map_err(|error| {
            MacroError::new(
                wgsl_source_span,
                format!("failed to translate shader to Rust: {}", ErrorChain(&error)),
            )
        })?;

    let translated_tokens: TokenStream = translated_source.parse().map_err(|error| {
        MacroError::new(
            wgsl_source_span,
            format!(
                "internal error: translator did not produce valid Rust: {}",
                ErrorChain(&error)
            ),
        )
    })?;

    Ok(translated_tokens)
}

// -------------------------------------------------------------------------------------------------

/// Formatting wrapper which prints an [`Error`] together with its `source()` chain.
///
/// The text begins with the [`fmt::Display`] format of the error.
#[derive(Clone, Copy, Debug)]
struct ErrorChain<'a>(&'a (dyn Error + 'a));

impl fmt::Display for ErrorChain<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_error_chain(fmt, self.0)
    }
}

fn format_error_chain(fmt: &mut fmt::Formatter<'_>, mut error: &(dyn Error + '_)) -> fmt::Result {
    write!(fmt, "{error}")?;
    while let Some(source) = error.source() {
        error = source;
        write!(fmt, "\n↳ {error}")?;
    }

    Ok(())
}
