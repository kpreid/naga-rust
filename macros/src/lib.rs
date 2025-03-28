//! This is a proc-macro helper library. Don't use this library directly; use [`naga_rust_embed`]
//! instead.
//!
//! [`naga_rust_embed`]: https://docs.rs/naga-rust-embed

#![allow(missing_docs, reason = "not intended to be used directly")]

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::PathBuf;

use quote::quote;
use syn::Token;

use naga_rust_back::Config;
use naga_rust_back::naga;

#[proc_macro]
pub fn include_wgsl_mr(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ConfigAndStr {
        config,
        string: path_literal,
    } = syn::parse_macro_input!(input as ConfigAndStr);

    match include_wgsl_mr_impl(config, &path_literal) {
        Ok(expansion) => expansion.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

#[proc_macro]
pub fn wgsl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ConfigAndStr {
        config,
        string: source_literal,
    } = syn::parse_macro_input!(input as ConfigAndStr);

    match parse_and_translate(config, source_literal.span(), &source_literal.value()) {
        Ok(expansion) => expansion.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Returns the input unchanged.
#[proc_macro_attribute]
pub fn dummy_attribute(
    _meta: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    input
}

// -------------------------------------------------------------------------------------------------

struct ConfigAndStr {
    config: Config,
    string: syn::LitStr,
}

impl syn::parse::Parse for ConfigAndStr {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let mut config = macro_default_config();
        loop {
            // Try parsing the final string literal.
            let not_a_string_error = match input.parse::<syn::LitStr>() {
                Ok(string) => {
                    // Accept a final optional comma after the string.
                    if !input.is_empty() {
                        input.parse::<Token![,]>()?;
                    }
                    return Ok(Self { config, string });
                }
                Err(e) => e,
            };

            let option_name = input.parse::<syn::Ident>().map_err(|mut e| {
                e.combine(not_a_string_error);
                e
            })?;
            input.parse::<Token![=]>()?;
            match &*option_name.to_string() {
                "global_struct" => {
                    config = config.global_struct(input.parse::<syn::Ident>()?.to_string());
                }
                // TODO: implement other configuration options
                _ => {
                    return Err(syn::Error::new_spanned(
                        option_name,
                        "unrecognized configuration option name",
                    ));
                }
            }
            input.parse::<Token![,]>()?;
        }
    }
}

fn macro_default_config() -> Config {
    Config::default().runtime_path("::naga_rust_embed::rt")
}

// -------------------------------------------------------------------------------------------------

fn include_wgsl_mr_impl(
    config: Config,
    path_literal: &syn::LitStr,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    // We use manifest-relative paths because currently, there is no way to arrange for
    // source-file-relative paths.
    let mut absolute_path: PathBuf = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    absolute_path.push(path_literal.value());

    // If this fails then we can't generate the `include_str!` we must generate.
    let absolute_path_str = absolute_path.to_str().ok_or_else(|| {
        syn::Error::new_spanned(
            path_literal,
            format_args!(
                "absolute path “{p:?}” must be UTF-8",
                p = absolute_path.display()
            ),
        )
    })?;

    let wgsl_source_text: String = fs::read_to_string(&absolute_path).map_err(|error| {
        syn::Error::new_spanned(
            path_literal,
            format_args!("failed to read “{absolute_path_str}”: {error}"),
        )
    })?;

    let translated_tokens = parse_and_translate(config, path_literal.span(), &wgsl_source_text)?;

    Ok(quote! {
        // Dummy include_str! call tells the compiler that we depend on this file,
        // which it would not notice otherwise.
        const _: &str = include_str!(#absolute_path_str);

        #translated_tokens
    })
}

fn parse_and_translate(
    config: Config,
    wgsl_source_span: proc_macro2::Span,
    wgsl_source_text: &str,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let module: naga::Module = naga::front::wgsl::parse_str(wgsl_source_text).map_err(|error| {
        syn::Error::new(
            wgsl_source_span,
            format_args!("failed to parse WGSL text: {}", ErrorChain(&error)),
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
        syn::Error::new(
            wgsl_source_span,
            format_args!("failed to validate WGSL: {}", ErrorChain(&error)),
        )
    })?;

    let translated_source: String = naga_rust_back::write_string(&module, &module_info, config)
        .map_err(|error| {
            syn::Error::new(
                wgsl_source_span,
                format_args!("failed to translate shader to Rust: {}", ErrorChain(&error)),
            )
        })?;

    let translated_tokens: proc_macro2::TokenStream =
        translated_source.parse().map_err(|error| {
            syn::Error::new(
                wgsl_source_span,
                format_args!(
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
