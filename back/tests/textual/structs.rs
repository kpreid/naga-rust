use pretty_assertions::assert_eq;

use naga_rust_back::{Condition, Config, Effect};

use crate::translate;

// -------------------------------------------------------------------------------------------------

#[test]
fn struct_decl_and_ctor() {
    assert_eq!(
        translate(
            Config::new(),
            r"struct Foo {
                x: i32,
                y: u32,
            }"
        ),
        indoc::indoc! {
            "#[::naga_rust_rt::derive(::naga_rust_rt::Clone, ::naga_rust_rt::Copy, ::naga_rust_rt::Debug, ::naga_rust_rt::PartialEq)]
            #[repr(C)]
            struct Foo {
                x: i32,
                y: u32,
            }
            impl Foo {
                fn new(
                    x: impl ::naga_rust_rt::Into<i32>,
                    y: impl ::naga_rust_rt::Into<u32>,
                ) -> Self {
                    Self { x: ::naga_rust_rt::into(x), y: ::naga_rust_rt::into(y) }
                }
            }
            "
        }
    );
}

#[test]
fn struct_custom_derive_rule() {
    assert_eq!(
        translate(
            Config::new().rule((
                Condition::Struct("Foo".to_owned()),
                Effect::Derive("bytemuck::NoUninit".to_owned())
            )),
            r"
            struct Foo { x: i32 }
            struct Bar { x: i32 }
            "
        ),
        indoc::indoc! {"
            #[::naga_rust_rt::derive(::naga_rust_rt::Clone, ::naga_rust_rt::Copy, ::naga_rust_rt::Debug, ::naga_rust_rt::PartialEq, bytemuck::NoUninit)]
            #[repr(C)]
            struct Foo {
                x: i32,
            }
            impl Foo {
                fn new(x: impl ::naga_rust_rt::Into<i32>) -> Self {
                    Self { x: ::naga_rust_rt::into(x) }
                }
            }
            #[::naga_rust_rt::derive(::naga_rust_rt::Clone, ::naga_rust_rt::Copy, ::naga_rust_rt::Debug, ::naga_rust_rt::PartialEq)]
            #[repr(C)]
            struct Bar {
                x: i32,
            }
            impl Bar {
                fn new(x: impl ::naga_rust_rt::Into<i32>) -> Self {
                    Self { x: ::naga_rust_rt::into(x) }
                }
            }
        "}
    );
}
