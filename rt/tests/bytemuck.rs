extern crate naga_rust_rt as rt;

use bytemuck::{AnyBitPattern, NoUninit, Zeroable};

#[test]
fn cast_from_scalar() {
    assert_eq!(
        bytemuck::must_cast::<rt::Scalar<f32>, f32>(rt::Scalar(1.0)),
        1.0
    );
}

#[test]
fn cast_to_scalar() {
    assert_eq!(
        bytemuck::must_cast::<f32, rt::Scalar<f32>>(1.0),
        rt::Scalar(1.0)
    );
}

#[test]
fn cast_from_vector() {
    assert_eq!(
        bytemuck::must_cast::<rt::Vec2<f32>, [f32; 2]>(rt::Vec2::new(1.0, 2.0)),
        [1.0, 2.0]
    );
}

#[test]
fn cast_to_vector() {
    assert_eq!(
        bytemuck::must_cast::<[f32; 2], rt::Vec2<f32>>([1.0, 2.0]),
        rt::Vec2::new(1.0, 2.0)
    );
}

#[test]
fn cast_from_matrix() {
    assert_eq!(
        bytemuck::must_cast::<rt::Mat2x2<f32>, [[f32; 2]; 2]>(rt::Mat2x2::new(
            rt::Vec2::new(1.0, 2.0),
            rt::Vec2::new(3.0, 4.0)
        )),
        [[1.0, 2.0], [3.0, 4.0]]
    );
}

#[test]
fn cast_to_matrix() {
    assert_eq!(
        bytemuck::must_cast::<[[f32; 2]; 2], rt::Mat2x2<f32>>([[1.0, 2.0], [3.0, 4.0]]),
        rt::Mat2x2::new(rt::Vec2::new(1.0, 2.0), rt::Vec2::new(3.0, 4.0))
    );
}

// Test that bytemuck traits are not implemented when not implementing them would be unsound.
// Note we don’t have 100% coverage for matrix types, but that’s okay because we use a macro to
// generate them consistently.
// Pointer types implement no traits:
static_assertions::assert_not_impl_any!(rt::Scalar<Box<f32>>: Zeroable, NoUninit, AnyBitPattern);
static_assertions::assert_not_impl_any!(rt::Vec2<Box<f32>>: Zeroable, NoUninit, AnyBitPattern);
static_assertions::assert_not_impl_any!(rt::Vec3<Box<f32>>: Zeroable, NoUninit, AnyBitPattern);
static_assertions::assert_not_impl_any!(rt::Vec4<Box<f32>>: Zeroable, NoUninit, AnyBitPattern);
static_assertions::assert_not_impl_any!(rt::Mat2x2<Box<f32>>: Zeroable, NoUninit, AnyBitPattern);
// If the element is not AnyBitPattern then neither is the container:
static_assertions::assert_not_impl_any!(rt::Scalar<bool>: AnyBitPattern);
static_assertions::assert_not_impl_any!(rt::Vec2<bool>: AnyBitPattern);
static_assertions::assert_not_impl_any!(rt::Vec3<bool>: AnyBitPattern);
static_assertions::assert_not_impl_any!(rt::Vec4<bool>: AnyBitPattern);
static_assertions::assert_not_impl_any!(rt::Mat2x2<bool>: AnyBitPattern);
