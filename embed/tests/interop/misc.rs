use naga_rust_embed::wgsl;

/// There is currently no option for directing the output of `debugPrintf` to a
/// testable destination, so this is only a “it doesn’t panic” smoke test for now.
#[test]
fn debug_printf() {
    // TODO: Add further testing of special characters after
    // <https://github.com/gfx-rs/wgpu/issues/10659> is fixed.

    wgsl! {r#"
        enable wgpu_debug_printf;
        fn f(value: u32) {
            debugPrintf("%% {Hello}, %u! %%", value);
        }
    "#};

    f(1234);
}

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
