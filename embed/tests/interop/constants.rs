use naga_rust_embed::wgsl;

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
