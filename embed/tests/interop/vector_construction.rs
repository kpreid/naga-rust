//! Vector constructors and swizzles.

use naga_rust_embed::wgsl;
use naga_rust_rt::{Vec2, Vec4};

#[test]
pub(crate) fn vector_mixed_construction() {
    wgsl!(
        r"fn foo() -> vec4f {
            return vec4f(1.0, vec2f(2.0, 3.0), 4.0);
        }"
    );

    assert_eq!(foo(), Vec4::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
pub(crate) fn vector_cast() {
    wgsl!(
        r"fn func(x: vec2f) -> vec2i {
            return vec2i(x);
        }"
    );

    // Expect truncation
    assert_eq!(func(Vec2::new(1.5, -1.5)), Vec2::new(1, -1));
}
