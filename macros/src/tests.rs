//! Tests of syntax errors on the input to our macros.
//!
//! These tests are not perfectly realistic, but they are much cheaper than invoking rustc and
//! require less setup.

use quote::quote;

use crate::ConfigAndStr;

// -------------------------------------------------------------------------------------------------

fn expect_error(input: proc_macro2::TokenStream) -> String {
    match syn::parse2::<ConfigAndStr>(input) {
        Ok(_) => panic!("no error"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn success_without_config() {
    let input = quote! { r#"foo("bar");"# };
    let parsed = syn::parse2::<ConfigAndStr>(input).unwrap();
    assert_eq!(parsed.string.value(), r#"foo("bar");"#);
}

#[test]
fn success_with_comma() {
    let input = quote! { r#"foo("bar");"#, };
    let parsed = syn::parse2::<ConfigAndStr>(input).unwrap();
    assert_eq!(parsed.string.value(), r#"foo("bar");"#);
}

#[test]
fn success_with_config() {
    let input = quote! { allow_unimplemented = true, r#"foo("bar");"#, };
    let parsed = syn::parse2::<ConfigAndStr>(input).unwrap();
    // TODO: add an escape hatch so we can check the result of config parsing here
    assert_eq!(parsed.string.value(), r#"foo("bar");"#);
}

#[test]
fn empty() {
    assert_eq!(
        expect_error(quote! {}),
        "unexpected end of input, expected identifier"
    );
}

#[test]
fn wrong_first_token() {
    assert_eq!(expect_error(quote! { ! }), "expected identifier");
}

#[test]
fn wrong_literal() {
    assert_eq!(
        expect_error(quote! { 3.0 }),
        // TODO: this is not a good error.
        "unexpected end of input, expected identifier"
    );
}

#[test]
fn unrecognized_config() {
    assert_eq!(
        expect_error(quote! { unknown_option = true, "" }),
        "`unknown_option` is not the name of a configuration option"
    );
}

#[test]
fn config_without_comma() {
    assert_eq!(
        expect_error(quote! { allow_unimplemented = true "" }),
        "expected `,`"
    );
}

#[test]
fn config_non_boolean() {
    assert_eq!(
        expect_error(quote! { allow_unimplemented = 3, "" }),
        "expected boolean literal"
    );
}

#[test]
fn config_non_ident() {
    assert_eq!(
        expect_error(quote! { global_struct = 3, "" }),
        "expected identifier"
    );
}

#[test]
fn non_comma_after_input() {
    assert_eq!(expect_error(quote! { ""+ }), "expected `,`");
}
