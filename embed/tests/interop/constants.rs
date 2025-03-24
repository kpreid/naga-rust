use naga_rust_embed::rt::Scalar;
use naga_rust_embed::wgsl;

#[test]
fn global_constant() {
    wgsl!(
        r"
        const X: f32 = 1234.0;
        fn get_x() -> f32 {
            return X;
        }
        "
    );
    // TODO: Exposing Scalar here is largely accidental. Should we really?
    assert_eq!(X, Scalar(1234.0));
    assert_eq!(get_x(), 1234.0);
}

#[test]
fn local_constant() {
    wgsl!(
        r"fn get_x() -> f32 {
            const X: f32 = 1234.0;
            return X;
        }"
    );
    assert_eq!(get_x(), 1234.0);
}
