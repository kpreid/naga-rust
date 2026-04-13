//! Tests which actually execute shader code and exercise interoperation with ordinary Rust.

mod arrays;
mod constants;
mod control_flow;
mod globals_and_functions;
mod math_functions;
mod matrices;
mod operators;
mod structs;
mod textures;
mod vector_construction;

// -------------------------------------------------------------------------------------------------

use naga_rust_embed::wgsl;

// TODO: implement atomics
// #[test]
// fn atomics() {
//     wgsl!(
//         global_struct = Globals,
//         r"
//         @group(0) @binding(0)
//         var<storage, read_write> storage_atomic_scalar: atomic<u32>;
//
//         fn atomic_ops() {
//             atomicAdd(&storage_atomic_scalar, 10u);
//             atomicAnd(&storage_atomic_scalar, 3u);
//         }
//         "
//     );
//
//     let globals = Globals {
//         x: core::sync::atomic::AtomicU32::new(2);
//     };
//     globals.atomic_ops();
//     assert_eq!(globals.x.load(core::sync::atomic::Ordering::Relaxed), 36);
// }

#[test]
fn allowed_unimplemented() {
    wgsl!(
        allow_unimplemented = true,
        r"
        fn example(x: f32) -> f32 {
            return dpdx(x);
        }
        "
    );

    let result = std::panic::catch_unwind(|| {
        example(1.0);
    });
    assert_eq!(
        result.unwrap_err().downcast_ref::<&str>().unwrap(),
        &"not implemented: this shader function contains a feature \
            which cannot yet be translated to Rust, derivatives"
    );
}
