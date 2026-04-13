use naga_rust_embed::rt::{Scalar, Vec4};
use naga_rust_embed::wgsl;

fn traits_implemented<T: Copy + core::fmt::Debug + PartialEq>() {}

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

    traits_implemented::<StructTest>();

    let mut s = StructTest { a: 1, b: 2.0 };
    modify_struct(&mut s);
    assert!(matches!(s, StructTest { a: 2, b: 3.0 }));
}

#[test]
fn ctor() {
    wgsl!(
        r"
        struct StructTest {
            a: i32,
            b: f32,
        }

        fn make_struct(a: i32, b: f32) -> StructTest {
            // in WGSL, all construction is done using function syntax
            return StructTest(a, b);
        }
        "
    );

    assert_eq!(make_struct(123, 456.0), StructTest { a: 123, b: 456.0 });
    // each such struct also has a Rust-style new() function, with `Into` flexibility
    assert_eq!(StructTest::new(123, 456.0), StructTest { a: 123, b: 456.0 });
    assert_eq!(
        StructTest::new(Scalar(123), Scalar(456.0)),
        StructTest { a: 123, b: 456.0 }
    );
}

/// This used to fail because struct fields weren’t being converted from Rust type `i32`
/// to Rust type `Scalar<i32>` and so the API was not as expected.
#[test]
fn struct_field_becomes_scalar() {
    wgsl!(
        r"
        struct StructTest {
            a: i32,
            b: f32,
        }
        fn combine(s: StructTest) -> f32 {
            return f32(s.a) + s.b;
        }
        "
    );

    assert_eq!(combine(StructTest { a: 123, b: 456.0 }), 579.0);
}

/// Test that we don't accidentally make `Scalar<Vec4<f32>>` out of a vector struct field.
#[test]
fn struct_field_already_vector() {
    wgsl!(
        r"
        struct StructTest {
            a: f32,
            b: vec4f,
        }
        fn combine(s: StructTest) -> vec4f {
            return s.b * s.a;
        }
        "
    );

    assert_eq!(
        combine(StructTest {
            a: 2.,
            b: Vec4::new(1., 2., 3., 4.)
        }),
        Vec4::new(2., 4., 6., 8.)
    );
}
