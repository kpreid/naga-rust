use naga_rust_embed::rt::Scalar;
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
