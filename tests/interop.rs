#![allow(missing_docs)]

// TODO: Should there should be an explicit public vector-type API module which is not rt::?
use naga_rust::rt::Vec2;
use naga_rust::wgsl;

#[test]
fn scalar_arithmetic() {
    wgsl!(
        r"fn add_one(x: i32) -> i32 {
            return x + 1;
        }"
    );

    assert_eq!(add_one(10), 11);
}

#[test]
fn vector_arithmetic() {
    wgsl!(
        r"fn add_one(x: vec2f) -> vec2f {
            return x + vec2f(1.0);
        }"
    );

    assert_eq!(add_one(Vec2::new(0.5, 10.0)), Vec2::new(1.5, 11.0));
}

#[test]
fn declare_and_modify_struct() {
    wgsl!(
        r"
        struct StructTest {
            a: i32,
            b: f32,
        }

        fn modify_struct(s_ptr: ptr<function, StructTest>) {
            (*s_ptr).a += 1;
            (*s_ptr).b += 1.0;
        }
        "
    );

    let mut s = StructTest { a: 1, b: 2.0 };
    modify_struct(&mut s);
    assert!(matches!(s, StructTest { a: 2, b: 3.0 }));
}
