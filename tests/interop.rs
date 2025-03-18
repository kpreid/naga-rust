#![allow(missing_docs)]

// TODO: Should there should be an explicit public vector-type API module which is not rt::?
use naga_rust::rt::{IVec2, Vec2};
use naga_rust::wgsl;

#[test]
fn global_constant() {
    wgsl!("const X: f32 = 1234.0;");
    assert_eq!(X, 1234.0);
}

#[test]
fn local_constant() {
    wgsl!(
        r"fn foo() -> f32 {
            const X: f32 = 1234.0;
            return X;
        }"
    );
    assert_eq!(foo(), 1234.0);
}

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
fn scalar_pointer() {
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
fn vector_pointer() {
    wgsl!(
        r"fn add_to_ptr(p: ptr<function, vec2i>) {
            (*p).x += 1;
            (*p).y += 2;
        }"
    );

    let mut x = IVec2::new(10, 10);
    add_to_ptr(&mut x);
    assert_eq!(x, IVec2::new(11, 12));
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

#[test]
fn switch() {
    wgsl!(
        r"fn switching(x: i32) -> i32 {
            switch (x) {
                case 0 { return 0; }
                case 1 { return 1; }
                case default { return 2; }
            }
        }"
    );

    assert_eq!(switching(0), 0);
    assert_eq!(switching(1), 1);
    assert_eq!(switching(2), 2);
}

#[test]
fn while_loop() {
    wgsl!(
        r"fn count(n: i32) -> i32 {
            var i: i32 = 0;
            while i < n {
                i += 1;
            }
            return i;
        }"
    );

    assert_eq!(count(10), 10);
}
