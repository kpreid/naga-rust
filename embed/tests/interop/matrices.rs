//! Tests for matrix operations.
//! Not very thorough yet.

use naga_rust_embed::rt::{self, Vec4};
use naga_rust_embed::wgsl;
use naga_rust_rt::{Mat4x4, Vec2};

#[test]
fn transpose() {
    wgsl!(
        r"
        fn t() -> mat4x4<f32> {
            let m = mat4x4<f32>(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            );
            return transpose(m); 
        }
        "
    );

    assert_eq!(
        t(),
        Mat4x4::new(
            Vec4::new(1.0, 5.0, 9.0, 13.0),
            Vec4::new(2.0, 6.0, 10.0, 14.0),
            Vec4::new(3.0, 7.0, 11.0, 15.0),
            Vec4::new(4.0, 8.0, 12.0, 16.0),
        )
    );
}

#[test]
#[rustfmt::skip]
fn matrix_vector_multiply() {
    wgsl!(
        r"
        fn mul_mat_vec(v: vec4f) -> vec4f {
            let m = mat4x4<f32>(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            );
            return m * v; 
        }

        fn mul_vec_mat(v: vec4f) -> vec4f {
            let m = mat4x4<f32>(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            );
            return v * m; 
        }
        "
    );

    assert_eq!(mul_mat_vec(Vec4::new(10.0, 0.0, 0.0, 0.0)), Vec4::new(10.0, 20.0, 30.0, 40.0));
    assert_eq!(mul_mat_vec(Vec4::new(0.0, 10.0, 0.0, 0.0)), Vec4::new(50.0, 60.0, 70.0, 80.0));
    assert_eq!(mul_mat_vec(Vec4::new(10.0, 10.0, 0.0, 0.0)), Vec4::new(60.0, 80.0, 100.0, 120.0));
    assert_eq!(mul_mat_vec(Vec4::new(0.0, 0.0, 100.0, 1.0)), Vec4::new(913.0, 1014.0, 1115.0, 1216.0));

    assert_eq!(mul_vec_mat(Vec4::new(10.0, 0.0, 0.0, 0.0)), Vec4::new(10.0, 50.0, 90.0, 130.0));
    assert_eq!(mul_vec_mat(Vec4::new(0.0, 10.0, 0.0, 0.0)), Vec4::new(20.0, 60.0, 100.0, 140.0));
    assert_eq!(mul_vec_mat(Vec4::new(10.0, 10.0, 0.0, 0.0)), Vec4::new(30.0, 110.0, 190.0, 270.0));
    assert_eq!(mul_vec_mat(Vec4::new(0.0, 0.0, 100.0, 1.0)), Vec4::new(304.0, 708.0, 1112.0, 1516.0));
}

#[test]
fn matrix_matrix_multiply() {
    wgsl!(
        r"
        fn mul() -> mat4x4<f32> {
            // example affine matrices (transposed so columns are textual columns)
            // translation
            let t = transpose(mat4x4<f32>(
                1.0, 0.0, 0.0, 1.0,
                0.0, 1.0, 0.0, 2.0,
                0.0, 0.0, 1.0, 3.0,
                0.0, 0.0, 0.0, 1.0,
            ));
            // scaling
            let s = transpose(mat4x4<f32>(
                2.0, 0.0, 0.0, 0.0,
                0.0, 2.0, 0.0, 0.0,
                0.0, 0.0, 2.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ));
            return s * t; 
        }
        "
    );

    assert_eq!(
        mul(),
        Mat4x4::new(
            Vec4::new(2.0, 0.0, 0.0, 2.0),
            Vec4::new(0.0, 2.0, 0.0, 4.0),
            Vec4::new(0.0, 0.0, 2.0, 6.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        )
        .transpose()
    );
}

#[test]
fn matrix_column_access() {
    wgsl!(
        r"
        fn matrix_columns(m: mat2x2f) -> array<vec2f, 2> {
            return array(m[0], m[1]);
        }
        "
    );

    let matrix = rt::Mat2x2::new(Vec2::new(1., 2.), Vec2::new(3., 4.));
    assert_eq!(
        matrix_columns(matrix),
        [Vec2::new(1., 2.), Vec2::new(3., 4.)]
    );
}
