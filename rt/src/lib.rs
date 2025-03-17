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

/// KLUDGE: `mutants::skip` is a no-op attribute macro, which we are borrowing to make our own
/// no-op attributes. This should be replaced with a real proc-macro lib.
pub use mutants::skip as addrspace;
pub use mutants::skip as binding;
pub use mutants::skip as compute;
pub use mutants::skip as fragment;
pub use mutants::skip as group;
pub use mutants::skip as invariant;
pub use mutants::skip as interpolate;
pub use mutants::skip as vertex;
pub use mutants::skip as workgroup_size;
