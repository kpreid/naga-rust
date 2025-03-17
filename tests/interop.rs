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
fn use_struct_definition_inline() {
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
