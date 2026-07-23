use proc_macro2::Delimiter;
use proc_macro2::Group;
use proc_macro2::Ident;
use proc_macro2::Punct;
use proc_macro2::Spacing;
use proc_macro2::Span;
use proc_macro2::TokenTree;

// -------------------------------------------------------------------------------------------------

/// Wrapper around a token stream iterator which provides help with parsing.
/// Performs a similar function to `syn::ParseStream`.
pub(crate) struct Parser {
    previous_token_span: Option<Span>,
    iter: proc_macro2::token_stream::IntoIter,
}

impl Parser {
    pub fn from_token_stream(token_stream: proc_macro2::TokenStream) -> Self {
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
        match unwrap_invisible_groups(self.next_expect("identifier")?) {
            TokenTree::Ident(ident) => Ok(ident.to_string()),
            other => Err(MacroError::unexpected_token(&other, "an identifier")),
        }
    }

    pub fn expect_bool(&mut self) -> Result<bool, MacroError> {
        let token = unwrap_invisible_groups(self.next_expect("a boolean literal")?);
        match litrs::BoolLit::try_from(&token) {
            Ok(lit) => Ok(lit.value()),
            Err(_) => Err(MacroError::unexpected_token(&token, "a boolean literal")),
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

        proc_macro2::TokenStream::from_iter(
            simple_path_to_tokens(span, &["core", "compile_error"])
                .chain([
                    TokenTree::Punct(Punct::new('!', Spacing::Alone)),
                    TokenTree::Group(Group::new(
                        Delimiter::Parenthesis,
                        proc_macro2::TokenStream::from(TokenTree::Literal(
                            proc_macro2::Literal::string(message),
                        )),
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
