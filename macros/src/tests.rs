//! Tests of syntax errors on the input to our macros.
//!
//! These tests are not perfectly realistic, but they are much cheaper than invoking rustc and
//! require less setup.

use quote::quote;

use crate::ConfigAndStr;

// -------------------------------------------------------------------------------------------------

fn expect_error(input: proc_macro2::TokenStream) -> String {
    ConfigAndStr::parse(input).unwrap_err().message
}

#[test]
fn success_without_config() {
    let input = quote! { r#"foo("bar");"# };
    let parsed = ConfigAndStr::parse(input).unwrap();
    assert_eq!(parsed.string, r#"foo("bar");"#);
}

#[test]
fn success_with_comma() {
    let input = quote! { r#"foo("bar");"#, };
    let parsed = ConfigAndStr::parse(input).unwrap();
    assert_eq!(parsed.string, r#"foo("bar");"#);
}

#[test]
fn success_with_config() {
    let input = quote! { allow_unimplemented = true, r#"foo("bar");"#, };
    let parsed = ConfigAndStr::parse(input).unwrap();
    // TODO: add an escape hatch so we can check the result of config parsing here
    assert_eq!(parsed.string, r#"foo("bar");"#);
}

#[test]
fn empty() {
    assert_eq!(
        expect_error(quote! {}),
        "expected a string literal or configuration option; found empty input"
    );
}

#[test]
fn wrong_first_token() {
    assert_eq!(
        expect_error(quote! { ! }),
        "expected a string literal or configuration option; found `!`"
    );
}

#[test]
fn wrong_literal() {
    assert_eq!(
        expect_error(quote! { 3.0 }),
        "expected a string literal or configuration option; found `3.0`"
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
        r#"expected comma; found `""`"#
    );
}

#[test]
fn config_non_boolean() {
    assert_eq!(
        expect_error(quote! { allow_unimplemented = 3, "" }),
        "expected a boolean literal; found `3`"
    );
}

#[test]
fn config_non_ident() {
    assert_eq!(
        expect_error(quote! { global_struct = 3, "" }),
        "expected an identifier; found `3`"
    );
}

#[test]
fn non_comma_after_input() {
    assert_eq!(
        expect_error(quote! { ""+ }),
        "expected comma or nothing; found `+`"
    );
}

mod rule_errors {
    use super::*;

    #[test]
    fn empty_rule() {
        assert_eq!(
            expect_error(quote! { rule(), "" }),
            "rule must have at least one effect"
        );
    }

    #[test]
    fn condition_but_no_effect() {
        assert_eq!(
            expect_error(quote! { rule(struct(Foo)), "" }),
            "rule must have at least one effect"
        );
    }

    #[test]
    fn empty_condition() {
        assert_eq!(
            expect_error(quote! { rule(struct()), "" }),
            "expected a struct name; found nothing after this"
        );
    }

    #[test]
    fn unknown_name() {
        assert_eq!(
            expect_error(quote! { rule(somethingorother()), "" }),
            "`somethingorother` is not the name of a rule condition or effect"
        );
    }
    #[test]
    fn no_arrow() {
        assert_eq!(
            expect_error(quote! { rule(struct(Foo) derive(PartialOrd)), "" }),
            "rule effects must be separated from conditions by `=>`"
        );
    }

    #[test]
    fn extra_arrow() {
        assert_eq!(
            expect_error(quote! { rule(struct(Foo) => derive(PartialOrd) => derive(Ord)), "" }),
            "rule must have at most one `=>`"
        );
    }

    #[test]
    fn empty_effect() {
        assert_eq!(
            expect_error(quote! { rule(struct(Foo) => derive()), "" }),
            "expected a path to a derive macro; found nothing"
        );
    }

    #[test]
    fn other_token() {
        assert_eq!(
            expect_error(quote! { rule(struct(Foo) => :), "" }),
            "expected a rule condition or effect; found `:`"
        );
    }

    #[test]
    fn path_colon_not_joint() {
        assert_eq!(
            expect_error(quote! { rule(derive(some : : trait)), "" }),
            "expected a path to a derive macro; found `:`"
        );
    }
}
