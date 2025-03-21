#![no_std]

mod vector;

pub use vector::*;

// TODO: should probably be num_traits::Zero or something custom
pub fn zero<T: Default>() -> T {
    T::default()
}

pub fn mix<T>(_v1: T, _v2: T, _a: f32) -> T {
    todo!()
}

pub fn discard() {
    // Best we can do for now, until we implement a codegen option to return Result instead.
    panic!("shader reached discard instruction");
}

pub use naga_rust_macros::dummy_attribute as compute;
pub use naga_rust_macros::dummy_attribute as fragment;
pub use naga_rust_macros::dummy_attribute as vertex;
pub use naga_rust_macros::dummy_attribute as workgroup_size;
