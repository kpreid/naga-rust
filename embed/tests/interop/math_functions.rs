//! Tests for built-in mathematical functions.

use naga_rust_embed::rt::Vec3;
use naga_rust_embed::wgsl;

// TODO: This test suite should include a test for every variant of `naga::MathFunction`,
// ideally checked for exhaustiveness, but that will be tricky.

// TODO: add allowance for floating point error in functions which permit it

macro_rules! math_function_test {
    // Ideally we would build shim code, but we can't pass `concat!()` to `wgsl!()`.
    ($test_name:ident, $code:literal, [$( ($case_args:tt, $case_result:expr) ),* $(,)?]) => {
        #[test]
        fn $test_name() {
            wgsl!($code);
            $(
                assert_eq!(shim $case_args, $case_result, concat!("case ", stringify!($case_args)));
            )*
        }
    };
}

math_function_test!(
    abs,
    "fn shim(x: f32) -> f32 { return abs(x); }",
    [((-1.0,), 1.0), ((0.0,), 0.0), ((1.0,), 1.0)]
);

// TODO: a* trig functions

math_function_test!(
    clamp_float,
    "fn shim(x: f32, min: f32, max: f32) -> f32 { return clamp(x, min, max); }",
    [
        ((0.0, 1.0, 2.0), 1.0),
        ((1.0, 1.0, 2.0), 1.0),
        ((2.0, 1.0, 2.0), 2.0),
        ((3.0, 1.0, 2.0), 2.0),
        ((0.0, 2.0, 1.0), 1.0),
        ((1.0, 2.0, 1.0), 1.0),
        ((2.0, 2.0, 1.0), 1.0),
        ((3.0, 2.0, 1.0), 1.0),
    ]
);

math_function_test!(
    ceil,
    "fn shim(x: f32) -> f32 { return ceil(x); }",
    [
        ((1.5,), 2.0),
        ((1.0,), 1.0),
        ((0.5,), 1.0),
        ((0.0,), 0.0),
        ((-0.5,), 0.0),
        ((-1.0,), -1.0),
        ((-1.5,), -1.0)
    ]
);

math_function_test!(
    clamp_integer,
    "fn shim(x: i32, min: i32, max: i32) -> i32 { return clamp(x, min, max); }",
    [
        ((0, 1, 2), 1),
        ((1, 1, 2), 1),
        ((2, 1, 2), 2),
        ((3, 1, 2), 2),
        ((0, 2, 1), 1),
        ((1, 2, 1), 1),
        ((2, 2, 1), 1),
        ((3, 2, 1), 1),
    ]
);

// TODO: cos, cosh
// TODO: count*
// TODO: cross
// TODO: degrees
// TODO: determinant

math_function_test!(
    distance,
    "fn shim(p: vec3f, q: vec3f) -> f32 { return distance(p, q); }",
    [(
        (Vec3::new(10.0, 20.0, 30.0), Vec3::new(11.0, 22.0, 33.0)),
        14.0_f32.sqrt()
    )]
);

// TODO: dot*
// TODO: exp
// TODO: extractBits
// TODO: faceForward
// TODO: first*
// TODO: floor
// TODO: fma
// TODO: fract
// TODO: frexp
// TODO: insertBits
// TODO: inverseSqrt
// TODO: ldexp
// TODO: length
// TODO: log, log2

math_function_test!(
    max_float,
    "fn shim(x: f32, y: f32) -> f32 { return max(x, y); }",
    [((1.0, 2.0), 2.0), ((2.0, 1.0), 2.0), ((1.0, 1.0), 1.0)]
);

math_function_test!(
    max_integer,
    "fn shim(x: i32, y: i32) -> i32 { return max(x, y); }",
    [((1, 2), 2), ((2, 1), 2), ((1, 1), 1)]
);

math_function_test!(
    min_float,
    "fn shim(x: f32, y: f32) -> f32 { return min(x, y); }",
    [((1.0, 2.0), 1.0), ((2.0, 1.0), 1.0), ((1.0, 1.0), 1.0)]
);

math_function_test!(
    min_integer,
    "fn shim(x: i32, y: i32) -> i32 { return min(x, y); }",
    [((1, 2), 1), ((2, 1), 1), ((1, 1), 1)]
);

// TODO: mix
// TODO: modf
// TODO: normalize
// TODO: MathFunction::Outer (not documented, not WGSL)
// TODO: pow
// TODO: quantizeToF16
// TODO: radians
// TODO: reflect
// TODO: refract
// TODO: sign
// TODO: sin, sinh

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

// TODO: sqrt
// TODO: step
// TODO: tan
// TODO: tanh
// TODO: transpose
// TODO: trunc
