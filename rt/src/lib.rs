mod construct;
mod vector;

pub use construct::New;
pub use glam::swizzles;
pub use vector::*;

pub fn new<T: New>(args: T::Args) -> T {
    T::new(args)
}

pub const fn select<T: Copy>(f: T, t: T, cond: bool) -> T {
    if cond { t } else { f }
}
pub fn select3(f: Vec3, t: Vec3, mask: BVec3) -> Vec3 {
    Vec3::select(mask, t, f)
}
pub fn select4(f: Vec4, t: Vec4, mask: BVec4) -> Vec4 {
    // TODO: resolve whether we are using mask types
    Vec4::select(BVec4A::from_array(mask.into()), t, f)
}

// TODO: should probably be num_traits::Zero or something custom
pub fn zero<T: Default>() -> T {
    T::default()
}

pub fn mix<T>(_v1: T, _v2: T, _a: f32) -> T {
    todo!()
}

pub use naga_rust_macros::dummy_attribute as addrspace;
pub use naga_rust_macros::dummy_attribute as binding;
pub use naga_rust_macros::dummy_attribute as compute;
pub use naga_rust_macros::dummy_attribute as fragment;
pub use naga_rust_macros::dummy_attribute as group;
pub use naga_rust_macros::dummy_attribute as invariant;
pub use naga_rust_macros::dummy_attribute as interpolate;
pub use naga_rust_macros::dummy_attribute as vertex;
pub use naga_rust_macros::dummy_attribute as workgroup_size;
