use proc_macro2::Span;
use proc_macro2::TokenStream;
use proc_macro2::TokenTree;

use naga_rust_back::Config;

use crate::parsing::{MacroError, Parser, unwrap_invisible_groups};

// -------------------------------------------------------------------------------------------------

/// Parsed syntax for the [`crate::wgsl`] or [`crate::include_wgsl_mr`] macros, which consist of
/// configuration options `name = value_expr` followed by a string literal which is either source
/// code or a path.
#[derive(Debug)]
pub(crate) struct ConfigAndStr {
    pub config: Config,
    pub string_span: Span,
    pub string: String,
}

impl ConfigAndStr {
    pub(crate) fn parse(input: TokenStream) -> Result<Self, MacroError> {
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

                    config = match &*option_name {
                        // The options parsed by this match should also be documented in
                        // `embed/src/configuration_syntax.md`.
                        // The ordering here is alphabetical.
                        "allow_unimplemented" => config.allow_unimplemented(input.expect_bool()?),
                        "explicit_types" => config.explicit_types(input.expect_bool()?),
                        "global_struct" => config.global_struct(input.expect_ident()?),
                        "include_functions" => config.include_functions(input.expect_bool()?),
                        "public_items" => config.public_items(input.expect_bool()?),
                        // TODO: raw_pointers doesn’t actually work, and will need to be marked unsafe
                        // when it is implemented. So, we don’t offer it yet.
                        //
                        // "raw_pointers" => {
                        //     config.raw_pointers(input.expect_bool()?)
                        // }
                        "resource_struct" => config.resource_struct(input.expect_ident()?),
                        _ => {
                            return Err(MacroError::new(
                                option_name_ident.span(),
                                format!(
                                    "`{option_name}` is not the name of a configuration option"
                                ),
                            ));
                        }
                    };

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

/// Returns the default configuration used by all shader translation macros.
fn macro_default_config() -> Config {
    Config::default()
        .runtime_path("::naga_rust_embed::rt")
        // Helps give better errors when the generated code is wrong.
        // TODO: Consider turning this back off for efficiency? Measure impact?
        .explicit_types(true)
}
