use naga_rust_embed::wgsl;

/// Functions can be called even if other functions in the same shader use unimplemented features.
#[test]
fn unimplemented_feature_not_called() {
    wgsl!(
        allow_unimplemented = true,
        "
        fn f() {
            storageBarrier();
        }
        fn g() -> i32 {
            return 1234;
        }
        "
    );

    assert_eq!(g(), 1234);
}

#[test]
#[should_panic = "not implemented: this shader function contains a feature which cannot yet be translated to Rust, barriers"]
fn unimplemented_feature_panic() {
    wgsl!(
        allow_unimplemented = true,
        "fn f() {
            storageBarrier();
        }"
    );

    f();
}
