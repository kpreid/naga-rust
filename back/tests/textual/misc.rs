//! Textual tests which aren’t like very many others and don’t have their own module.

use pretty_assertions::assert_eq;

use naga_rust_back::Config;

use crate::{translate, translate_body};

// -------------------------------------------------------------------------------------------------

const DOCUMENTATION_INPUT: &str = r"
    //! Module documentation (not translated because our output has no module).

    /// Struct documentation.
    /// Second line of struct documentation.
    ///
    /// Second paragraph of struct documentation. The previous line has no space.
    struct DStruct {
        /// Field documentation.
        field: i32,
    }

    /// Constant documentation.
    const D_CONSTANT: i32 = 1;

    /// Global variable documentation.
    var<private> d_variable: i32 = 2;

    /// Uniform documentation.
    @group(1) @binding(2) var<uniform> d_uniform: i32;

    /// Function documentation.
    fn d_func() {}

    /// Entry point documentation.
    @fragment
    fn d_entry(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
        return vec4f(1.0);
    }
";

#[test]
fn documentation_enabled() {
    assert_eq!(
        translate(
            Config::new()
                .global_struct("Globals")
                .resource_struct("Resources")
                .include_documentation(true),
            DOCUMENTATION_INPUT
        ),
        indoc::indoc! {r#"
            #[doc = "Struct documentation."]
            #[doc = "Second line of struct documentation."]
            #[doc = ""]
            #[doc = "Second paragraph of struct documentation. The previous line has no space."]
            #[::naga_rust_rt::derive(::naga_rust_rt::Clone, ::naga_rust_rt::Copy, ::naga_rust_rt::Debug, ::naga_rust_rt::PartialEq)]
            #[repr(C)]
            struct DStruct {
                #[doc = "Field documentation."]
                field: i32,
            }
            impl DStruct {
                fn new(field: impl ::naga_rust_rt::Into<i32>) -> Self {
                    Self { field: ::naga_rust_rt::into(field) }
                }
            }
            #[doc = "Constant documentation."]
            #[allow(non_upper_case_globals)]
            const D_CONSTANT: ::naga_rust_rt::Scalar<i32> = ::naga_rust_rt::Scalar(1i32);
            struct Resources {
                #[doc = "Uniform documentation."]
                #[doc = ""]
                #[doc = "group(1) binding(2)"]
                d_uniform: ::naga_rust_rt::Scalar<i32>,
            }
            struct Globals<'g> {
                resources: &'g Resources,
                #[doc = "Global variable documentation."]
                d_variable: ::naga_rust_rt::Scalar<i32>,
            }
            impl<'g> Globals<'g> {
                const fn new(resources: &'g Resources) -> Self {
                    Self { resources, d_variable: ::naga_rust_rt::Scalar(2i32) }
                }
            }
            impl<'g> Globals<'g> {
                #[doc = "Function documentation."]
                fn d_func(&self) {
                    ::naga_rust_rt::into(self.v_d_func())
                }
                #[doc = "Function documentation."]
                #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
                fn v_d_func(&self) {
                    return;
                }
                #[doc = "Entry point documentation."]
                fn d_entry(&self, position: impl ::naga_rust_rt::Into<::naga_rust_rt::Vec4<f32>>) -> ::naga_rust_rt::Vec4<f32> {
                    ::naga_rust_rt::into(self.v_d_entry(::naga_rust_rt::into(position)))
                }
                #[doc = "Entry point documentation."]
                #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
                fn v_d_entry(&self, position: ::naga_rust_rt::Vec4<f32>) -> ::naga_rust_rt::Vec4<f32> {
                    return ::naga_rust_rt::Vec4::splat_from_scalar(::naga_rust_rt::Scalar(1f32));
                }
            }
        "#}
    );
}

#[test]
fn documentation_disabled() {
    assert_eq!(
        translate(
            Config::new()
                .global_struct("Globals")
                .resource_struct("Resources")
                .include_documentation(false),
            DOCUMENTATION_INPUT
        ),
        indoc::indoc! {r#"
            #[::naga_rust_rt::derive(::naga_rust_rt::Clone, ::naga_rust_rt::Copy, ::naga_rust_rt::Debug, ::naga_rust_rt::PartialEq)]
            #[repr(C)]
            struct DStruct {
                field: i32,
            }
            impl DStruct {
                fn new(field: impl ::naga_rust_rt::Into<i32>) -> Self {
                    Self { field: ::naga_rust_rt::into(field) }
                }
            }
            #[allow(non_upper_case_globals)]
            const D_CONSTANT: ::naga_rust_rt::Scalar<i32> = ::naga_rust_rt::Scalar(1i32);
            struct Resources {
                #[doc = "group(1) binding(2)"]
                d_uniform: ::naga_rust_rt::Scalar<i32>,
            }
            struct Globals<'g> {
                resources: &'g Resources,
                d_variable: ::naga_rust_rt::Scalar<i32>,
            }
            impl<'g> Globals<'g> {
                const fn new(resources: &'g Resources) -> Self {
                    Self { resources, d_variable: ::naga_rust_rt::Scalar(2i32) }
                }
            }
            impl<'g> Globals<'g> {
                fn d_func(&self) {
                    ::naga_rust_rt::into(self.v_d_func())
                }
                #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
                fn v_d_func(&self) {
                    return;
                }
                fn d_entry(&self, position: impl ::naga_rust_rt::Into<::naga_rust_rt::Vec4<f32>>) -> ::naga_rust_rt::Vec4<f32> {
                    ::naga_rust_rt::into(self.v_d_entry(::naga_rust_rt::into(position)))
                }
                #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
                fn v_d_entry(&self, position: ::naga_rust_rt::Vec4<f32>) -> ::naga_rust_rt::Vec4<f32> {
                    return ::naga_rust_rt::Vec4::splat_from_scalar(::naga_rust_rt::Scalar(1f32));
                }
            }
        "#}
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
            "#[::naga_rust_rt::derive(::naga_rust_rt::Clone, ::naga_rust_rt::Copy, ::naga_rust_rt::Debug, ::naga_rust_rt::PartialEq)]
            #[repr(C)]
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
            struct Resources<'g> {
                #[doc = \"group(0) binding(1)\"]
                arr: &'g [::naga_rust_rt::Scalar<u32>],
            }
            impl<'g> Resources<'g> {
                fn length(&self) -> u32 {
                    ::naga_rust_rt::into(self.v_length())
                }
                #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
                fn v_length(&self) -> ::naga_rust_rt::Scalar<u32> {
                    return ::naga_rust_rt::array_length(self.arr);
                }
            }
            "
        }
    );
}

/// Interim test for atomic types while we don't support atomic statements.
///
/// TODO: This test is wrong because the atomic scalar should not be `&mut`, but further
/// work will be needed to implement that.
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
            struct Resources<'g> {
                #[doc = \"group(0) binding(0)\"]
                atomic_scalar: &'g mut ::core::sync::atomic::AtomicU32,
            }
            impl<'g> Resources<'g> {
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
        translate_body(
            Config::new(),
            r"fn f(p: ptr<private, array<i32, 4>>) -> i32 {
                return ~(*p)[2];
            }"
        ),
        indoc::indoc! {"{
            let _e2 = (*p)[2usize];
            return (!_e2);
        }"}
    );
}
