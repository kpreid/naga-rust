//! Tests of the exact source text produced by the Rust backend.
//!
//! TODO: Consider whether we want to rewrite some or all of these tests to use
//! `syn` or another parser to match trees instead of exact text.

use core::error::Error as ErrorTrait;
use core::fmt;

use pretty_assertions::assert_eq;

use naga_rust_back::Config;

fn translate(config: Config, wgsl_source_text: &str) -> String {
    fn inner(config: Config, wgsl_source_text: &str) -> Result<String, Box<dyn ErrorTrait>> {
        let module: naga::Module = naga::front::wgsl::parse_str(wgsl_source_text)?;

        let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&module)?;

        let translated_source: String =
            naga_rust_back::write_string(&module, &module_info, config)?;

        Ok(translated_source)
    }

    match inner(config, wgsl_source_text) {
        Ok(translated_source) => translated_source,
        Err(e) => panic!("{}", ErrorChain(&*e)),
    }
}

#[track_caller]
fn expect_error(config: Config, wgsl_source_text: &str) -> naga_rust_back::Error {
    let module: naga::Module = naga::front::wgsl::parse_str(wgsl_source_text).unwrap();
    let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module)
    .unwrap();

    match naga_rust_back::write_string(&module, &module_info, config) {
        Ok(_) => panic!("expected error, but got success"),
        Err(e) => e,
    }
}

#[track_caller]
fn expect_unimplemented(wgsl_source_text: &str) {
    match expect_error(Config::default(), wgsl_source_text) {
        naga_rust_back::Error::Unimplemented(_) => {}
        e => panic!(
            "expected Error::Unimplemented, but got a different error:\n{}",
            ErrorChain(&e)
        ),
    }
}

// -------------------------------------------------------------------------------------------------

#[test]
fn visibility_control() {
    assert_eq!(
        translate(Config::new(), "fn foo() {}"),
        indoc::indoc! {
            r"
            fn foo() {
                v_foo().into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_foo() {
                return;
            }
            
            "
        }
    );
    assert_eq!(
        translate(Config::new().public_items(true), "fn foo() {}"),
        indoc::indoc! {
            r"
            pub fn foo() {
                v_foo().into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_foo() {
                return;
            }
            
            "
        }
    );
}

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
        indoc::indoc! {
            r"
            #[::naga_rust_rt::fragment]
            fn main(position: impl ::naga_rust_rt::Into<::naga_rust_rt::Vec4<f32>>) -> ::naga_rust_rt::Vec4<f32> {
                v_main(position.into()).into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_main(position: ::naga_rust_rt::Vec4<f32>) -> ::naga_rust_rt::Vec4<f32> {
                return ::naga_rust_rt::Vec4::splat_from_scalar(::naga_rust_rt::Scalar(1f32));
            }
            "
        }
    );
}

#[test]
fn global_variable_enabled() {
    assert_eq!(
        translate(
            Config::new().global_struct("Globals"),
            r"var<private> foo: i32 = 1;"
        ),
        indoc::indoc! {
            "
            struct Globals {
                foo: ::naga_rust_rt::Scalar<i32>,
            }
            impl Default for Globals {
                fn default() -> Self { Self {
                    foo: ::naga_rust_rt::Scalar(1i32),
                }}
            }
            impl Globals {
            }
            "
        }
    );
}
#[test]
fn global_variable_disabled() {
    assert!(matches!(
        expect_error(Config::new(), r"var<private> foo: i32 = 1;"),
        naga_rust_back::Error::GlobalVariablesNotEnabled { example: _, .. }
    ));
}

#[test]
fn switch() {
    assert_eq!(
        translate(
            Config::new(),
            r"
            fn switching(x: i32) -> i32 {
                switch (x) {
                    case 0 { return 0; }
                    case 1, 2 { return 1; }
                    case default { return 2; }
                }
            }
            "
        ),
        indoc::indoc! {
            "
            fn switching(x: impl ::naga_rust_rt::Into<::naga_rust_rt::Scalar<i32>>) -> i32 {
                v_switching(x.into()).into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_switching(x: ::naga_rust_rt::Scalar<i32>) -> ::naga_rust_rt::Scalar<i32> {
                match ::naga_rust_rt::Scalar::into_inner(x) {
                    0i32 => {
                        return ::naga_rust_rt::Scalar(0i32);
                    }
                    1i32 | 2i32 => {
                        return ::naga_rust_rt::Scalar(1i32);
                    }
                    _ => {
                        return ::naga_rust_rt::Scalar(2i32);
                    }
                }
            }

            "
        }
    );
}

#[test]
fn array_type_sizes() {
    assert_eq!(
        translate(
            Config::new(),
            r"struct Foo {
                x: array<i32, 10>,
                y: array<i32>,
            }"
        ),
        indoc::indoc! {
            "#[repr(C)]
            struct Foo {
                x: [i32; 10],
                y: [i32],
            }

            "
        }
    );
}

#[test]
fn array_length() {
    assert_eq!(
        translate(
            Config::new().global_struct("Globals"),
            r"
            @group(0) @binding(1) var<storage> arr: array<u32>;
            fn length() -> u32 {
                return arrayLength(&arr);
            }
            ",
        ),
        // TODO: we don't yet fully support bindings properly so lots of this code is nonsense.
        // This test is only intending to check the translation of arrayLength(), which is
        // hard to test separately since it must take a `ptr<storage, array<..>>`.
        //
        indoc::indoc! {
            "
            struct Globals {
                // group(0) binding(1)
                arr: [::naga_rust_rt::Scalar<u32>],
            }
            impl Default for Globals {
                fn default() -> Self { Self {
                    arr: Default::default(),
                }}
            }
            impl Globals {
            fn length(&self, ) -> u32 {
                self.v_length().into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_length(&self, ) -> ::naga_rust_rt::Scalar<u32> {
                return (&self.arr).len();
            }

            }
            "
        }
    );
}

/// Interim test for atomic types while we don't support atomic statements.
#[test]
fn atomic_type() {
    assert_eq!(
        translate(
            Config::new().global_struct("Globals"),
            r"
            @group(0) @binding(0)
            var<storage, read_write> atomic_scalar: atomic<u32>;
            ",
        ),
        indoc::indoc! {
            "
            struct Globals {
                // group(0) binding(0)
                atomic_scalar: ::core::sync::atomic::AtomicU32,
            }
            impl Default for Globals {
                fn default() -> Self { Self {
                    atomic_scalar: Default::default(),
                }}
            }
            impl Globals {
            }
            "
        }
    );
}

/// Verify that the output is not ignorant of parentheses needed for precedence
/// by combining prefix and postfix operators.
#[test]
fn precedence_of_prefix_and_postfix() {
    assert_eq!(
        translate(
            Config::new(),
            r"fn f(p: ptr<private, array<i32, 4>>) -> i32 {
                return ~(*p)[2];
            }"
        ),
        indoc::indoc! {
            "
            fn f(p: &mut [::naga_rust_rt::Scalar<i32>; 4]) -> i32 {
                v_f(p).into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_f(p: &mut [::naga_rust_rt::Scalar<i32>; 4]) -> ::naga_rust_rt::Scalar<i32> {
                let _e2 = (*p)[2 as usize];
                return (!_e2);
            }

            "
        }
    );
}

#[test]
fn unimplemented_continuing() {
    expect_unimplemented(
        r"fn foo() { 
            var i = 0;
            loop {
                i += 1;
                continuing { i += 1; }
            }
        }",
    );
}

#[test]
fn unimplemented_break_if() {
    expect_unimplemented(
        r"fn foo() { 
            var i = 0;
            loop {
                continuing {
                    i += 1;
                    break if i > 10;
                }
            }
        }",
    );
}

// -------------------------------------------------------------------------------------------------

/// Formatting wrapper which prints an [`Error`] together with its `source()` chain.
///
/// We bother to do this for tests because it is way more legible than `unwrap()`'s Debug format.
/// Note that the same code exists in `naga-rust-macros` for user facing error reporting.
#[derive(Clone, Copy, Debug)]
struct ErrorChain<'a>(&'a (dyn ErrorTrait + 'a));

impl fmt::Display for ErrorChain<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_error_chain(fmt, self.0)
    }
}

fn format_error_chain(
    fmt: &mut fmt::Formatter<'_>,
    mut error: &(dyn ErrorTrait + '_),
) -> fmt::Result {
    write!(fmt, "{error}")?;
    while let Some(source) = error.source() {
        error = source;
        write!(fmt, "\n↳ {error}")?;
    }

    Ok(())
}
