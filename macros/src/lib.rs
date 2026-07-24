//! This is a proc-macro helper library. Don't use this library directly; use [`naga_rust_embed`]
//! instead.
//!
//! [`naga_rust_embed`]: https://docs.rs/naga-rust-embed

#![allow(missing_docs, reason = "not intended to be used directly")]

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::PathBuf;

use proc_macro2::Delimiter;
use proc_macro2::Group;
use proc_macro2::Ident;
use proc_macro2::Literal;
use proc_macro2::Punct;
use proc_macro2::Spacing;
use proc_macro2::Span;
use proc_macro2::TokenStream;
use proc_macro2::TokenTree;

use naga_rust_back::Config;
use naga_rust_back::naga;

// -------------------------------------------------------------------------------------------------

mod parsing;
use parsing::{MacroError, Parser, simple_path_to_tokens, unwrap_invisible_groups};

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

/// Returns the input unchanged.
#[proc_macro_attribute]
pub fn dummy_attribute(
    _meta: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    input
}

// -------------------------------------------------------------------------------------------------

/// Parsed syntax for the [`wgsl`] or [`include_wgsl_mr`] macros, which consist of configuration
/// options `name = value_expr` followed by a string literal which is either source code or a path.
#[derive(Debug)]
struct ConfigAndStr {
    config: Config,
    string_span: Span,
    string: String,
}

impl ConfigAndStr {
    fn parse(input: TokenStream) -> Result<Self, MacroError> {
        const EXPECT_TOP_LEVEL: &str = "a string literal or configuration option";
        let mut config = macro_default_config();
        let mut input = Parser::from_token_stream(input);
        loop {
            match unwrap_invisible_groups(input.next_expect(EXPECT_TOP_LEVEL)?) {
                // A literal must be the final string.
                ref tt @ TokenTree::Literal(ref literal_token) => {
                    let quoted: String = literal_token.to_string();
                    let unquoted: String = match litrs::StringLit::try_from(literal_token) {
                        Ok(sl) => sl.into_value(),
                        Err(e) => {
                            return Err(if quoted.starts_with('"') {
                                // It's probably a string literal but doesn’t parse.
                                MacroError::new(literal_token.span(), e.to_string())
                            } else {
                                // It's probably a non-string literal.
                                // Use our own error message so that we mention the possibility
                                // of a configuration option.
                                MacroError::unexpected_token(tt, EXPECT_TOP_LEVEL)
                            });
                        }
                    };

                    // Accept a final optional comma after the string.
                    match input.next() {
                        Some(TokenTree::Punct(punct)) if punct.as_char() == ',' => {}
                        None => {}
                        Some(other) => {
                            return Err(MacroError::unexpected_token(&other, "comma or nothing"));
                        }
                    }

                    return Ok(Self {
                        config,
                        string_span: literal_token.span(),
                        string: unquoted,
                    });
                }

                // An identifier must be the name of a configuration option.
                TokenTree::Ident(option_name_ident) => {
                    let option_name = option_name_ident.to_string();

                    match input.next_expect("`=`")? {
                        TokenTree::Punct(punct) if punct.as_char() == '=' => {}
                        other => {
                            return Err(MacroError::unexpected_token(&other, "`=`"));
                        }
                    }

                    match &*option_name {
                        // The options parsed by this match should also be documented in
                        // `embed/src/configuration_syntax.md`.
                        // The ordering here is alphabetical.
                        "allow_unimplemented" => {
                            config = config.allow_unimplemented(input.expect_bool()?);
                        }
                        "explicit_types" => {
                            config = config.explicit_types(input.expect_bool()?);
                        }
                        "global_struct" => {
                            config = config.global_struct(input.expect_ident()?);
                        }
                        "include_functions" => {
                            config = config.include_functions(input.expect_bool()?);
                        }
                        "public_items" => {
                            config = config.public_items(input.expect_bool()?);
                        }
                        // TODO: raw_pointers doesn’t actually work, and will need to be marked unsafe
                        // when it is implemented. So, we don’t offer it yet.
                        //
                        // "raw_pointers" => {
                        //     config = config.raw_pointers(input.expect_bool()?);
                        // }
                        "resource_struct" => {
                            config = config.resource_struct(input.expect_ident()?);
                        }
                        _ => {
                            return Err(MacroError::new(
                                option_name_ident.span(),
                                format!(
                                    "`{option_name}` is not the name of a configuration option"
                                ),
                            ));
                        }
                    }

                    match input.next_expect("comma")? {
                        TokenTree::Punct(punct) if punct.as_char() == ',' => {}
                        other => {
                            return Err(MacroError::unexpected_token(&other, "comma"));
                        }
                    }
                }

                other => {
                    return Err(MacroError::unexpected_token(&other, EXPECT_TOP_LEVEL));
                }
            }
        }
    }
}

fn macro_default_config() -> Config {
    Config::default()
        .runtime_path("::naga_rust_embed::rt")
        // Helps give better errors when the generated code is wrong.
        // TODO: Consider turning this back off for efficiency? Measure impact?
        .explicit_types(true)
}

// -------------------------------------------------------------------------------------------------

fn include_wgsl_mr_impl(
    config: Config,
    path_span: Span,
    path_text: &str,
) -> Result<TokenStream, MacroError> {
    // We use manifest-relative paths because currently, there is no way to arrange for
    // source-file-relative paths.
    let mut absolute_path: PathBuf = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    absolute_path.push(path_text);

    // If this fails then we can't generate the `include_str!` we must generate.
    let absolute_path_str = absolute_path.to_str().ok_or_else(|| {
        MacroError::new(
            path_span,
            format!(
                "absolute path “{p:?}” must be UTF-8",
                p = absolute_path.display()
            ),
        )
    })?;

    let wgsl_source_text: String = fs::read_to_string(&absolute_path).map_err(|error| {
        MacroError::new(
            path_span,
            format!("failed to read “{absolute_path_str}”: {error}"),
        )
    })?;

    let translated_tokens = parse_and_translate(config, path_span, &wgsl_source_text)?;

    // Dummy include_str! call tells the compiler that we depend on this file,
    // which it would not notice otherwise.
    let generated_span = Span::mixed_site(); // ideally would be def_site
    Ok(TokenStream::from_iter(
        [
            TokenTree::Ident(Ident::new("const", generated_span)),
            TokenTree::Ident(Ident::new("_", generated_span)),
            TokenTree::Punct(Punct::new(':', Spacing::Alone)),
            TokenTree::Punct(Punct::new('&', Spacing::Alone)),
        ]
        .into_iter()
        .chain(simple_path_to_tokens(
            generated_span,
            &["core", "primitive", "str"],
        ))
        .chain([TokenTree::Punct(Punct::new('=', Spacing::Alone))])
        .chain(simple_path_to_tokens(
            generated_span,
            &["core", "include_str"],
        ))
        .chain([
            TokenTree::Punct(Punct::new('!', Spacing::Alone)),
            TokenTree::Group(Group::new(
                Delimiter::Parenthesis,
                TokenStream::from(TokenTree::Literal({
                    let mut lit = Literal::string(absolute_path_str);
                    lit.set_span(path_span);
                    lit
                })),
            )),
            TokenTree::Punct(Punct::new(';', Spacing::Alone)),
        ])
        .chain(translated_tokens),
    ))
}

fn parse_and_translate(
    config: Config,
    wgsl_source_span: Span,
    wgsl_source_text: &str,
) -> Result<TokenStream, MacroError> {
    let module: naga::Module = naga::front::wgsl::parse_str(wgsl_source_text).map_err(|error| {
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
