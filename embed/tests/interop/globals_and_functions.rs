//! Tests of `global_struct` or its absence, and of function calls from outside and inside.

use naga_rust_embed::rt::{Scalar, Vec2, Vec4};
use naga_rust_embed::wgsl;

#[test]
fn call_entry_point() {
    wgsl!(
        r"@fragment
        fn frag(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
            return position + 1.0;
        }"
    );
    assert_eq!(
        frag(Vec4::new(1.0, 2.0, 0.0, 0.0)),
        Vec4::new(2.0, 3.0, 1.0, 1.0)
    );
}

#[test]
pub(crate) fn function_call() {
    wgsl!(
        r"
        fn func0() -> i32 {
            return 10;
        }
        fn func1(x: i32) -> i32 {
            return x * 3;
        }
        fn func2(x: i32, y: i32) -> i32 {
            return x + y;
        }
        fn funcs() -> i32 {
            return func0() + func1(2) + func2(1000, 6000);
        }
        "
    );

    assert_eq!(funcs(), 7016);
}

/// When `global_struct` is in use, calls to other functions in the shader must use
/// `self` syntax.
#[test]
pub(crate) fn function_call_with_self() {
    wgsl!(
        global_struct = Globals,
        r"
        var<private> foo: i32 = 1234;
        fn inner() -> i32 {
            return foo;
        }
        fn outer() -> i32 {
            return inner();
        }
        "
    );

    assert_eq!(Globals::new().outer(), 1234);
}

#[test]
pub(crate) fn uniform_binding() {
    wgsl!(
        resource_struct = Resources,
        r"
        struct Uniforms {
            @location(0) x: f32,
            @location(1) y: f32,
        };

        @group(0) @binding(0) var<uniform> ub: Uniforms;

        fn main() -> vec2<f32> {
            return vec2(ub.x, ub.y);
        }
    "
    );

    assert_eq!(
        Resources {
            ub: Uniforms { x: 1.0, y: 2.0 }
        }
        .main(),
        Vec2::new(1.0, 2.0)
    )
}

#[test]
fn both_globals_and_resources() {
    wgsl!(
        global_struct = Globals,
        resource_struct = Resources,
        r"
        @group(0) @binding(0) var<uniform> foo: i32;
        var<private> bar: i32 = 1;
        fn combine() -> i32 {
            return foo + bar;
        } 
        "
    );

    assert_eq!(Globals::new(&Resources { foo: Scalar(100) }).combine(), 101);
}
