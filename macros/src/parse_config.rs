use naga_rust_back::Condition;
use naga_rust_back::Effect;
use naga_rust_back::Rule;
use proc_macro2::Spacing;
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

                    input.expect_eof("nothing")?;

                    return Ok(Self {
                        config,
                        string_span: literal_token.span(),
                        string: unquoted,
                    });
                }

                // An identifier must be the name of a configuration option.
                TokenTree::Ident(option_name_ident) => {
                    let option_name = option_name_ident.to_string();

                    config = match &*option_name {
                        // The options parsed by this match should also be documented in
                        // `embed/src/configuration_syntax.md`.
                        // The ordering here is alphabetical.
                        "allow_unimplemented" => {
                            config.allow_unimplemented(input.expect_eq()?.expect_bool()?)
                        }
                        "explicit_types" => {
                            config.explicit_types(input.expect_eq()?.expect_bool()?)
                        }
                        "global_struct" => config.global_struct(input.expect_eq()?.expect_ident()?),
                        "include_functions" => {
                            config.include_functions(input.expect_eq()?.expect_bool()?)
                        }
                        "public_items" => config.public_items(input.expect_eq()?.expect_bool()?),
                        // TODO: raw_pointers doesn’t actually work, and will need to be marked unsafe
                        // when it is implemented. So, we don’t offer it yet.
                        //
                        // "raw_pointers" => {
                        //     config.raw_pointers(input.expect_bool()?)
                        // }
                        "resource_struct" => {
                            config.resource_struct(input.expect_eq()?.expect_ident()?)
                        }
                        "rule" => config.rule(parse_rule(&mut input)?),
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

// -------------------------------------------------------------------------------------------------

fn parse_rule(input: &mut Parser) -> Result<Rule, MacroError> {
    // The grammar of a rule is
    //   (condition '=>')? effect*
    // but for simplicity and error reporting, the implementation is more like
    //   (condition | effect | '=>')*
    // with separate validation of the ordering.

    // A condition or an effect.
    // TODO: Better name than "term"?
    enum Term {
        Condition(Condition),
        Effect(Effect),
    }

    let mut input = input.expect_parenthesis()?;

    let mut conditions = Vec::new();
    let mut effects = Vec::new();
    let mut seen_arrow = false;

    'rule: loop {
        match input.next() {
            Some(TokenTree::Ident(term_ident)) => {
                let term_ident_string = term_ident.to_string();
                let mut term_args = input.expect_parenthesis()?;

                let term_value = match &*term_ident_string {
                    "derive" => Term::Effect(Effect::Derive(parse_path(
                        &mut term_args,
                        "a path to a derive macro",
                    )?)),
                    "struct" => Term::Condition(Condition::Struct(
                        term_args.expect_ident_tok("a struct name")?.to_string(),
                    )),
                    _ => {
                        return Err(MacroError::new(
                            term_ident.span(),
                            format!("`{term_ident}` is not the name of a rule condition or effect"),
                        ));
                    }
                };

                match term_value {
                    Term::Condition(condition) => {
                        if seen_arrow {
                            return Err(MacroError::new(
                                term_ident.span(),
                                "rule conditions must come before the `=>`".to_string(),
                            ));
                        }
                        conditions.push(condition);
                    }
                    Term::Effect(effect) => {
                        if !conditions.is_empty() && !seen_arrow {
                            return Err(MacroError::new(
                                term_ident.span(),
                                "rule effects must be separated from conditions by `=>`"
                                    .to_string(),
                            ));
                        }
                        effects.push(effect);
                    }
                }
            }
            Some(ref arrow_tok @ TokenTree::Punct(ref punct)) if punct.as_char() == '=' => {
                input.expect_punct('>', "`=>`")?;
                if seen_arrow {
                    return Err(MacroError::new(
                        arrow_tok.span(),
                        "rule must have at most one `=>`".to_string(),
                    ));
                } else {
                    seen_arrow = true;
                }
            }
            None => {
                if effects.is_empty() {
                    return Err(MacroError::new(
                        input.previous_token_span.unwrap(),
                        "rule must have at least one effect".to_string(),
                    ));
                } else {
                    break 'rule;
                }
            }
            Some(other) => {
                return Err(MacroError::unexpected_token(
                    &other,
                    "a rule condition or effect",
                ));
            }
        }
    }

    Ok(Rule {
        conditions,
        effects,
    })
}

fn parse_path(input: &mut Parser, description: &'static str) -> Result<String, MacroError> {
    let mut path = String::new();

    loop {
        match input.next() {
            Some(TokenTree::Ident(ident)) => {
                path.push_str(&ident.to_string());
            }
            Some(TokenTree::Punct(punct))
                if punct.as_char() == ':' && punct.spacing() == Spacing::Joint =>
            {
                // If we found one colon, there must be a second colon.
                input.expect_punct(':', "a path separator `::`")?;
                path.push_str("::");
            }
            Some(TokenTree::Punct(ref punct)) if punct.as_char() == '<' => {
                return Err(MacroError::new(
                    punct.span(),
                    "paths with parameters are not yet supported".to_string(),
                ));
            }
            None => {
                break;
            }
            Some(other) => {
                return Err(MacroError::unexpected_token(&other, description));
            }
        }
    }

    if path.is_empty() {
        return Err(MacroError::new(
            input.previous_token_span.unwrap_or_else(Span::call_site),
            format!("expected {description}; found nothing"),
        ));
    }

    Ok(path)
}
