use exhaust::Exhaust as _;

use naga_rust_embed::wgsl;
use naga_rust_rt::{Vec2, Vec4};

#[test]
pub(crate) fn scalar_arithmetic() {
    wgsl!(
        r"fn add_one(x: i32) -> i32 {
            return x + 1;
        }"
    );

    assert_eq!(add_one(10), 11);
}

#[test]
pub(crate) fn vector_arithmetic() {
    wgsl!(
        r"fn add_one(x: vec2f) -> vec2f {
            return x + vec2f(1.0);
        }"
    );

    assert_eq!(add_one(Vec2::new(0.5, 10.0)), Vec2::new(1.5, 11.0));
}

#[test]
pub(crate) fn scalar_pointer() {
    wgsl!(
        r"fn add_one_ptr(p: ptr<function, i32>) {
            *p += 1;
        }"
    );

    let mut x = 10;
    add_one_ptr(&mut x);
    assert_eq!(x, 11);
}

#[test]
pub(crate) fn vector_pointer() {
    wgsl!(
        r"fn add_to_ptr(p: ptr<function, vec2i>) {
            (*p).x += 1;
            (*p).y += 2;
        }"
    );

    let mut x = Vec2::new(10, 10);
    add_to_ptr(&mut x);
    assert_eq!(x, Vec2::new(11, 12));
}

#[test]
pub(crate) fn bool_ops() {
    wgsl!(
        r"
        fn short_circuit(a: bool, b: bool, c: bool) -> bool {
            return a && b || c;
        }
        fn non_short_circuit(a: bool, b: bool, c: bool) -> bool {
            return a & b | c;
        }
        "
    );

    for [a, b, c] in <[bool; 3]>::exhaust() {
        assert_eq!(short_circuit(a, b, c), a && b || c);
        assert_eq!(non_short_circuit(a, b, c), a && b || c);
    }
}

#[test]
pub(crate) fn bool_vector_ops() {
    wgsl!(
        r"fn bool_vector_func(a: vec4<bool>, b: vec4<bool>, c: vec4<bool>) -> vec4<bool> {
            return a & b | c;
        }"
    );

    for [a, b, c] in <[[bool; 4]; 3]>::exhaust() {
        for (i, element) in <[bool; 4]>::from(bool_vector_func(a.into(), b.into(), c.into()))
            .into_iter()
            .enumerate()
        {
            assert_eq!(element, a[i] && b[i] || c[i]);
        }
    }
}

#[test]
pub(crate) fn comparison_of_scalars() {
    wgsl!(
        r"
        fn le(a: f32, b: f32) -> bool {
            return a <= b;
        }
        "
    );

    assert_eq!(le(1.0, 2.0), true);
    assert_eq!(le(2.0, 2.0), true);
    assert_eq!(le(3.0, 2.0), false);
}

#[test]
pub(crate) fn comparison_of_vectors() {
    wgsl!(
        r"
        fn le(a: vec4f, b: vec4f) -> vec4<bool> {
            return a <= b;
        }
        "
    );

    assert_eq!(
        le(Vec4::new(1.0, 2.0, 3.0, 4.0), Vec4::splat(2.0)),
        Vec4::new(true, true, false, false),
    )
}
