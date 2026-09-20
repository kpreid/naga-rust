//! Textual tests which aren’t like very many others and don’t have their own module.

use pretty_assertions::assert_eq;

use naga_rust_back::Config;

use crate::{translate, translate_body};

// -------------------------------------------------------------------------------------------------

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
