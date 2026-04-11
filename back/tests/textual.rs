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
            r"
            var<private> foo: i32 = 1;
            fn get_global() -> i32 { return foo; }
            "
        ),
        indoc::indoc! {
            "
            struct Globals {
                foo: ::naga_rust_rt::Scalar<i32>,
            }
            impl Globals {
                const fn new() -> Self { Self {
                    foo: ::naga_rust_rt::Scalar(1i32),
                }}
            }
            impl Default for Globals { fn default() -> Self { Self::new() } }
            impl Globals {
            fn get_global(&self, ) -> i32 {
                self.v_get_global().into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_get_global(&self, ) -> ::naga_rust_rt::Scalar<i32> {
                let _e1 = self.foo;
                return _e1;
            }

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
fn resources_enabled() {
    assert_eq!(
        translate(
            Config::new().resource_struct("Resources"),
            r"
            @group(0) @binding(0) var<uniform> foo: i32;
            fn get_uniform() -> i32 { return foo; }
            "
        ),
        indoc::indoc! {
            "
            struct Resources {
                // group(0) binding(0)
                foo: ::naga_rust_rt::Scalar<i32>,
            }
            impl Resources {
            fn get_uniform(&self, ) -> i32 {
                self.v_get_uniform().into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_get_uniform(&self, ) -> ::naga_rust_rt::Scalar<i32> {
                let _e1 = self.foo;
                return _e1;
            }

            }
            "
        }
    );
}

#[test]
fn resources_disabled() {
    assert!(matches!(
        expect_error(
            Config::new(),
            r"@group(0) @binding(0) var<uniform> foo: i32;"
        ),
        naga_rust_back::Error::ResourcesNotEnabled { example: _, .. }
    ));
}

/// Code generated when both `global_struct` and `resource_struct` are set.
///
/// This test also tests the `public_items` option, because that affects globals and functions.
#[test]
fn globals_and_resources_enabled_and_visibility() {
    let source = r"
        @group(0) @binding(0) var<uniform> foo: i32;
        var<private> bar: i32 = 1;
        fn combine() -> i32 {
            return foo + bar;
        } 
    ";

    // Without public items
    assert_eq!(
        translate(
            Config::new()
                .global_struct("Globals")
                .resource_struct("Resources"),
            source
        ),
        indoc::indoc! {
            "
            struct Resources {
                // group(0) binding(0)
                foo: ::naga_rust_rt::Scalar<i32>,
            }
            struct Globals<'g> {
                resources: &'g Resources,
                bar: ::naga_rust_rt::Scalar<i32>,
            }
            impl<'g> Globals<'g> {
                const fn new(resources: &'g Resources) -> Self { Self {
                    resources,
                    bar: ::naga_rust_rt::Scalar(1i32),
                }}
            }
            impl<'g> Globals<'g> {
            fn combine(&self, ) -> i32 {
                self.v_combine().into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_combine(&self, ) -> ::naga_rust_rt::Scalar<i32> {
                let _e1 = self.resources.foo;
                let _e3 = self.bar;
                return (_e1 + _e3);
            }

            }
            "
        }
    );

    // With public items
    assert_eq!(
        translate(
            Config::new()
                .global_struct("Globals")
                .resource_struct("Resources")
                .public_items(true),
            source
        ),
        indoc::indoc! {
            "
            struct Resources {
                // group(0) binding(0)
                pub foo: ::naga_rust_rt::Scalar<i32>,
            }
            struct Globals<'g> {
                pub resources: &'g Resources,
                pub bar: ::naga_rust_rt::Scalar<i32>,
            }
            impl<'g> Globals<'g> {
                pub const fn new(resources: &'g Resources) -> Self { Self {
                    resources,
                    bar: ::naga_rust_rt::Scalar(1i32),
                }}
            }
            impl<'g> Globals<'g> {
            pub fn combine(&self, ) -> i32 {
                self.v_combine().into()
            }
            #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
            fn v_combine(&self, ) -> ::naga_rust_rt::Scalar<i32> {
                let _e1 = self.resources.foo;
                let _e3 = self.bar;
                return (_e1 + _e3);
            }

            }
            "
        }
    );
}

#[test]
fn globals_and_resources_enabled_but_empty() {
    assert_eq!(
        translate(
            Config::new()
                .global_struct("Globals")
                .resource_struct("Resources"),
            r""
        ),
        indoc::indoc! {
            "
            struct Resources {
            }
            struct Globals<'g> {
                resources: &'g Resources,
            }
            impl<'g> Globals<'g> {
                const fn new(resources: &'g Resources) -> Self { Self {
                    resources,
                }}
            }
            impl<'g> Globals<'g> {
            }
            "
        }
    );
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

/// This test is only intending to check the translation of `arrayLength()`,
/// but it needs a resource to be able to get a `ptr<storage, array<..>>`.
#[test]
fn array_length() {
    assert_eq!(
        translate(
            Config::new().resource_struct("Resources"),
            r"
            @group(0) @binding(1) var<storage> arr: array<u32>;
            fn length() -> u32 {
                return arrayLength(&arr);
            }
            ",
        ),
        indoc::indoc! {
            "
            struct Resources {
                // group(0) binding(1)
                arr: [::naga_rust_rt::Scalar<u32>],
            }
            impl Resources {
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
            Config::new().resource_struct("Resources"),
            r"
            @group(0) @binding(0)
            var<storage, read_write> atomic_scalar: atomic<u32>;
            ",
        ),
        indoc::indoc! {
            "
            struct Resources {
                // group(0) binding(0)
                atomic_scalar: ::core::sync::atomic::AtomicU32,
            }
            impl Resources {
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
