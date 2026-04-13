//! Tests for built-in mathematical functions.

use naga_rust_embed::wgsl;

// TODO: This test suite should include a test for every variant of `naga::MathFunction`,
// ideally checked for exhaustiveness.

macro_rules! math_function_test {
    // Ideally we would build shim code, but we can't pass `concat!()` to `wgsl!()`.
    ($function_name:ident, $code:literal, [$( ($case_args:tt, $case_result:expr) ),* $(,)?]) => {
        #[test]
        fn $function_name() {
            wgsl!($code);
            $(
                assert_eq!(shim $case_args, $case_result, concat!("case ", stringify!($case_args)));
            )*
        }
    };
}

math_function_test!(
    smoothstep,
    "fn shim(x: f32, edge0: f32, edge1: f32) -> f32 { return smoothstep(x, edge0, edge1); }",
    [
        ((0.0, 10.0, 20.0), 0.0),
        ((10.0, 10.0, 20.0), 0.0),
        ((15.0, 10.0, 20.0), 0.5),
        ((20.0, 10.0, 20.0), 1.0),
        ((30.0, 10.0, 20.0), 1.0),
    ]
);
