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
