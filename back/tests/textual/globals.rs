use pretty_assertions::assert_eq;

use naga_rust_back::Config;

use crate::{expect_error, translate};

// -------------------------------------------------------------------------------------------------

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
                const fn new() -> Self {
                    Self { foo: ::naga_rust_rt::Scalar(1i32) }
                }
            }
            impl ::naga_rust_rt::Default for Globals {
                fn default() -> Self {
                    <Self>::new()
                }
            }
            impl Globals {
                fn get_global(&self) -> i32 {
                    ::naga_rust_rt::into(self.v_get_global())
                }
                #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
                fn v_get_global(&self) -> ::naga_rust_rt::Scalar<i32> {
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
    // TODO: When MSRV ≥ 1.87, use `assert_matches!`
    assert!(matches!(
        expect_error(Config::new(), r"var<private> foo: i32 = 1;"),
        naga_rust_back::Error::GlobalVariablesNotEnabled { example, .. }
        if example == "foo"
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
                #[doc = \"group(0) binding(0)\"]
                foo: ::naga_rust_rt::Scalar<i32>,
            }
            impl Resources {
                fn get_uniform(&self) -> i32 {
                    ::naga_rust_rt::into(self.v_get_uniform())
                }
                #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
                fn v_get_uniform(&self) -> ::naga_rust_rt::Scalar<i32> {
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
    // TODO: When MSRV ≥ 1.87, use `assert_matches!`
    assert!(matches!(
        expect_error(
            Config::new(),
            r"@group(0) @binding(0) var<uniform> foo: i32;"
        ),
        naga_rust_back::Error::ResourcesNotEnabled { example, .. }
        if example == "foo"
    ));
}

/// Code generated when both `global_struct` and `resource_struct` are set.
///
/// This test also tests the `public_items` option, because that affects globals and functions.
#[test]
fn globals_and_resources_enabled_and_visibility() {
    let source = r"
        @group(0) @binding(0) var<uniform> foo: i32;
        @group(0) @binding(1) var texture: texture_2d<f32>;
        var<private> bar: i32 = 1;
        const A_CONSTANT: i32 = 2;
        fn combine(baz: i32, qux: i32) -> i32 {
            return foo + bar + baz + qux;
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
        indoc::indoc! {"
            #[allow(non_upper_case_globals)]
            const A_CONSTANT: ::naga_rust_rt::Scalar<i32> = ::naga_rust_rt::Scalar(2i32);
            struct Resources<'g> {
                #[doc = \"group(0) binding(0)\"]
                foo: ::naga_rust_rt::Scalar<i32>,
                #[doc = \"group(0) binding(1)\"]
                texture: ::naga_rust_rt::texture::Texture2d<&'g dyn ::naga_rust_rt::texture::Read<Coordinates = ::naga_rust_rt::Vec2<i32>, Component = f32>>,
            }
            struct Globals<'g> {
                resources: &'g Resources<'g>,
                bar: ::naga_rust_rt::Scalar<i32>,
            }
            impl<'g> Globals<'g> {
                const fn new(resources: &'g Resources<'g>) -> Self {
                    Self { resources, bar: ::naga_rust_rt::Scalar(1i32) }
                }
            }
            impl<'g> Globals<'g> {
                fn combine(
                    &self,
                    baz: impl ::naga_rust_rt::Into<::naga_rust_rt::Scalar<i32>>,
                    qux: impl ::naga_rust_rt::Into<::naga_rust_rt::Scalar<i32>>,
                ) -> i32 {
                    ::naga_rust_rt::into(self.v_combine(::naga_rust_rt::into(baz), ::naga_rust_rt::into(qux)))
                }
                #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
                fn v_combine(
                    &self,
                    baz: ::naga_rust_rt::Scalar<i32>,
                    qux: ::naga_rust_rt::Scalar<i32>,
                ) -> ::naga_rust_rt::Scalar<i32> {
                    let _e3 = self.resources.foo;
                    let _e5 = self.bar;
                    return (((_e3 + _e5) + baz) + qux);
                }
            }
        "}
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
        indoc::indoc! {"
            #[allow(non_upper_case_globals)]
            pub const A_CONSTANT: ::naga_rust_rt::Scalar<i32> = ::naga_rust_rt::Scalar(2i32);
            pub struct Resources<'g> {
                #[doc = \"group(0) binding(0)\"]
                pub foo: ::naga_rust_rt::Scalar<i32>,
                #[doc = \"group(0) binding(1)\"]
                pub texture: ::naga_rust_rt::texture::Texture2d<&'g dyn ::naga_rust_rt::texture::Read<Coordinates = ::naga_rust_rt::Vec2<i32>, Component = f32>>,
            }
            pub struct Globals<'g> {
                pub resources: &'g Resources<'g>,
                pub bar: ::naga_rust_rt::Scalar<i32>,
            }
            impl<'g> Globals<'g> {
                pub const fn new(resources: &'g Resources<'g>) -> Self {
                    Self { resources, bar: ::naga_rust_rt::Scalar(1i32) }
                }
            }
            impl<'g> Globals<'g> {
                pub fn combine(
                    &self,
                    baz: impl ::naga_rust_rt::Into<::naga_rust_rt::Scalar<i32>>,
                    qux: impl ::naga_rust_rt::Into<::naga_rust_rt::Scalar<i32>>,
                ) -> i32 {
                    ::naga_rust_rt::into(self.v_combine(::naga_rust_rt::into(baz), ::naga_rust_rt::into(qux)))
                }
                #[allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)]
                fn v_combine(
                    &self,
                    baz: ::naga_rust_rt::Scalar<i32>,
                    qux: ::naga_rust_rt::Scalar<i32>,
                ) -> ::naga_rust_rt::Scalar<i32> {
                    let _e3 = self.resources.foo;
                    let _e5 = self.bar;
                    return (((_e3 + _e5) + baz) + qux);
                }
            }
        "}
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
                const fn new(resources: &'g Resources) -> Self {
                    Self { resources }
                }
            }
            impl<'g> Globals<'g> {
            }
            "
        }
    );
}
