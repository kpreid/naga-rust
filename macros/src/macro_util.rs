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

// -------------------------------------------------------------------------------------------------

/// Wrapper around a token stream iterator which provides help with parsing.
/// Performs a similar function to `syn::ParseStream`.
pub(crate) struct Parser {
    pub previous_token_span: Option<Span>,
    iter: proc_macro2::token_stream::IntoIter,
}

impl Parser {
    pub fn from_token_stream(token_stream: TokenStream) -> Self {
        Self {
            previous_token_span: None,
            iter: token_stream.into_iter(),
        }
    }

    /// Consume the next token, if there is one.
    pub fn next(&mut self) -> Option<TokenTree> {
        self.iter
            .next()
            .inspect(|token| self.previous_token_span = Some(token.span()))
    }

    /// Consume the next token, and return an error if there isn’t one.
    pub fn next_expect(&mut self, expected_thing: &'static str) -> Result<TokenTree, MacroError> {
        match self.next() {
            Some(token) => Ok(token),
            None => Err(if let Some(span) = self.previous_token_span {
                MacroError::new(
                    span,
                    format!("expected {expected_thing}; found nothing after this"),
                )
            } else {
                MacroError::new(
                    Span::call_site(),
                    format!("expected {expected_thing}; found empty input"),
                )
            }),
        }
    }

    // Note: The following parsing methods are inflexible in that they don’t take an expected_thing
    // nor return a span. This is just because we don’t have uses for those, not because they
    // wouldn’t make sense if we did.

    pub fn expect_ident(&mut self) -> Result<String, MacroError> {
        match unwrap_invisible_groups(self.next_expect("an identifier")?) {
            TokenTree::Ident(ident) => Ok(ident.to_string()),
            other => Err(MacroError::unexpected_token(&other, "an identifier")),
        }
    }

    /// Like `expect_ident` but returning the token and accepting a description
    pub fn expect_ident_tok(&mut self, description: &'static str) -> Result<Ident, MacroError> {
        match unwrap_invisible_groups(self.next_expect(description)?) {
            TokenTree::Ident(ident) => Ok(ident),
            other => Err(MacroError::unexpected_token(&other, description)),
        }
    }

    pub fn expect_bool(&mut self) -> Result<bool, MacroError> {
        let token = unwrap_invisible_groups(self.next_expect("a boolean literal")?);
        match litrs::BoolLit::try_from(&token) {
            Ok(lit) => Ok(lit.value()),
            Err(_) => Err(MacroError::unexpected_token(&token, "a boolean literal")),
        }
    }

    /// Consume a specific punctuation token token or error, then return `self` for chaining.
    pub fn expect_punct(
        &mut self,
        ch: char,
        description: &'static str,
    ) -> Result<&mut Self, MacroError> {
        match self.next_expect(description)? {
            TokenTree::Punct(punct) if punct.as_char() == ch => Ok(self),
            other => Err(MacroError::unexpected_token(&other, description)),
        }
    }

    /// Consume a `=` token or error, then return `self` for chaining.
    pub fn expect_eq(&mut self) -> Result<&mut Self, MacroError> {
        self.expect_punct('=', "`=`")
    }

    pub fn expect_parenthesis(&mut self) -> Result<Parser, MacroError> {
        match self.next_expect("parenthesis")? {
            TokenTree::Group(group) if group.delimiter() == Delimiter::Parenthesis => Ok(Parser {
                previous_token_span: Some(group.span_open()),
                iter: group.stream().into_iter(),
            }),
            other => Err(MacroError::unexpected_token(&other, "parenthesis")),
        }
    }

    pub(crate) fn expect_eof(&mut self, description: &'static str) -> Result<(), MacroError> {
        match self.next() {
            Some(other) => Err(MacroError::unexpected_token(&other, description)),
            None => Ok(()),
        }
    }
}

/// Removes invisible groups, which may occur if our input tokens were partially produced by a
/// `macro_rules!` macro.
pub fn unwrap_invisible_groups(mut token: TokenTree) -> TokenTree {
    loop {
        match token {
            TokenTree::Group(ref group) if group.delimiter() == Delimiter::None => {
                let mut it = group.stream().into_iter();
                let Some(inner_token) = it.next() else {
                    // no inner token
                    return token;
                };
                if it.next().is_some() {
                    // more than 1 inner token
                    return token;
                }
                token = inner_token;
            }
            _ => break token,
        }
    }
}

// -------------------------------------------------------------------------------------------------

/// Error that can be converted to a [`compile_error!`] invocation.
#[derive(Debug)]
pub(crate) struct MacroError {
    pub span: Span,
    pub message: String,
}

impl MacroError {
    pub fn new(span: Span, message: String) -> Self {
        Self { span, message }
    }

    pub fn unexpected_token(found: &TokenTree, expected_thing: &'static str) -> Self {
        match found {
            TokenTree::Group(group) => {
                // Special error case to not print the entire group, which might be long.
                Self::new(
                    group.span(),
                    format!("expected {expected_thing}; found group"),
                )
            }
            other => Self::new(
                other.span(),
                format!("expected {expected_thing}; found `{other}`"),
            ),
        }
    }

    pub fn to_compile_error(&self) -> proc_macro::TokenStream {
        // TODO: There are no tests which demonstrate that this does its job properly.
        let Self { span, ref message } = *self;

        TokenStream::from_iter(
            simple_path_to_tokens(span, &["core", "compile_error"])
                .chain([
                    TokenTree::Punct(Punct::new('!', Spacing::Alone)),
                    TokenTree::Group(Group::new(
                        Delimiter::Parenthesis,
                        TokenStream::from(TokenTree::Literal(Literal::string(message))),
                    )),
                    TokenTree::Punct(Punct::new(';', Spacing::Alone)),
                ])
                .map(|mut token| {
                    token.set_span(span);
                    token
                }),
        )
        .into()
    }
}

// -------------------------------------------------------------------------------------------------

pub(crate) fn simple_path_to_tokens(span: Span, path: &[&str]) -> impl Iterator<Item = TokenTree> {
    path.iter().flat_map(move |segment| {
        [
            TokenTree::Punct(Punct::new(':', Spacing::Joint)),
            TokenTree::Punct(Punct::new(':', Spacing::Alone)),
            TokenTree::Ident(Ident::new(segment, span)),
        ]
    })
}

// -------------------------------------------------------------------------------------------------

/// Read a file at a path which, if relative, is interpreted relative to the location of
/// `Cargo.toml`, and generate a matching `include_str!` call to tell the compiler that
/// the file is a dependency (since there is no proc-macro API to achieve that).
pub(crate) fn read_file_manifest_relative(
    path_span: Span,
    path_text: &str,
) -> Result<(String, TokenStream), MacroError> {
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

    let file_contents: String = fs::read_to_string(&absolute_path).map_err(|error| {
        MacroError::new(
            path_span,
            format!("failed to read “{absolute_path_str}”: {error}"),
        )
    })?;

    // Dummy include_str! call tells the compiler that we depend on this file,
    // which it would not notice otherwise.
    let generated_span = Span::mixed_site(); // ideally would be def_site
    let include_str_call = TokenStream::from_iter(
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
        ]),
    );

    Ok((file_contents, include_str_call))
}
