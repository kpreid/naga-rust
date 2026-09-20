//! Textual tests which aren’t like very many others and don’t have their own module.

use pretty_assertions::assert_eq;

use naga_rust_back::{Condition, Config, Effect, Inline};

use crate::translate;

// -------------------------------------------------------------------------------------------------

/// Checks handling of entry point functions, which are a separate input from normal functions.
#[test]
fn entry_point() {
    assert_eq!(
        translate(
            Config::new(),
            r"
            @fragment
            fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
                return vec4f(1.0);
            }"
        ),
        indoc::indoc! {r"
            fn main(position: impl ::naga_rust_rt::Into<::naga_rust_rt::Vec4<f32>>) -> ::naga_rust_rt::Vec4<f32> {
                ::naga_rust_rt::into(v_main(::naga_rust_rt::into(position)))
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_main(position: ::naga_rust_rt::Vec4<f32>) -> ::naga_rust_rt::Vec4<f32> {
                return ::naga_rust_rt::Vec4::splat_from_scalar(::naga_rust_rt::Scalar(1f32));
            }
        "}
    );
}

/// Compares output with `include_functions` on and off.
#[test]
fn omitting_functions() {
    let source = r"
        struct Struct { field: f32 }
        const CONSTANT: f32 = 1.0;
        fn function() {}
    ";
    let output_without_functions = indoc::indoc! {r"
        #[::naga_rust_rt::derive(::naga_rust_rt::Clone, ::naga_rust_rt::Copy, ::naga_rust_rt::Debug, ::naga_rust_rt::PartialEq)]
        #[repr(C)]
        struct Struct {
            field: f32,
        }
        impl Struct {
            fn new(field: impl ::naga_rust_rt::Into<f32>) -> Self {
                Self { field: ::naga_rust_rt::into(field) }
            }
        }
        #[allow(non_upper_case_globals)]
        const CONSTANT: ::naga_rust_rt::Scalar<f32> = ::naga_rust_rt::Scalar(1f32);
    "};
    let output_with_functions = output_without_functions.to_owned()
        + indoc::indoc! {r"
            fn function() {
                ::naga_rust_rt::into(v_function())
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_function() {
                return;
            }
        "};
    assert_eq!(
        translate(Config::new().include_functions(true), source),
        output_with_functions
    );
    assert_eq!(
        translate(Config::new().include_functions(false), source),
        output_without_functions
    );
}

/// If `include_functions` is disabled, then the checks for a missing `global_struct`
/// should also be disabled.
#[test]
fn omitting_functions_also_allows_omitting_globals() {
    assert_eq!(
        translate(
            Config::new().include_functions(false),
            r"
            const FOO_INIT: i32 = 1;
            var<private> foo: i32 = FOO_INIT;
            fn get_global() -> i32 { return foo; }
            "
        ),
        indoc::indoc! {"
            #[allow(non_upper_case_globals)]
            const FOO_INIT: ::naga_rust_rt::Scalar<i32> = ::naga_rust_rt::Scalar(1i32);
        "}
    );
}

/// If `include_functions` is disabled, then the checks for a missing `resource_struct`
/// should also be disabled.
#[test]
fn omitting_functions_also_allows_omitting_resources() {
    assert_eq!(
        translate(
            Config::new().include_functions(false),
            r"
            struct Uniforms {
                foo: i32,
            }
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            fn get_uniform() -> i32 { return uniforms.foo; }
            "
        ),
        indoc::indoc! {"
            #[::naga_rust_rt::derive(::naga_rust_rt::Clone, ::naga_rust_rt::Copy, ::naga_rust_rt::Debug, ::naga_rust_rt::PartialEq)]
            #[repr(C)]
            struct Uniforms {
                foo: i32,
            }
            impl Uniforms {
                fn new(foo: impl ::naga_rust_rt::Into<i32>) -> Self {
                    Self { foo: ::naga_rust_rt::into(foo) }
                }
            }
        "}
    );
}

#[test]
fn function_inlining_rule() {
    assert_eq!(
        translate(
            Config::new()
                .rule((
                    Condition::Function("has_inline".to_owned()),
                    Effect::Inline(Inline::Maybe)
                ))
                .rule((
                    Condition::Function("has_inline_always".to_owned()),
                    Effect::Inline(Inline::Always)
                ))
                .rule((
                    Condition::Function("has_inline_never".to_owned()),
                    Effect::Inline(Inline::Never)
                )),
            r"
            fn has_none() {}
            fn has_inline() {}
            fn has_inline_always() {}
            fn has_inline_never() {}
            "
        ),
        indoc::indoc! {"
            fn has_none() {
                ::naga_rust_rt::into(v_has_none())
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_has_none() {
                return;
            }
            #[inline]
            fn has_inline() {
                ::naga_rust_rt::into(v_has_inline())
            }
            #[inline]
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_has_inline() {
                return;
            }
            #[inline(always)]
            fn has_inline_always() {
                ::naga_rust_rt::into(v_has_inline_always())
            }
            #[inline(always)]
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_has_inline_always() {
                return;
            }
            #[inline(never)]
            fn has_inline_never() {
                ::naga_rust_rt::into(v_has_inline_never())
            }
            #[inline(never)]
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_has_inline_never() {
                return;
            }
        "}
    );
}
