//! Tests which actually execute shader code and exercise interoperation with ordinary Rust.

mod arrays;
mod constants;
mod control_flow;
mod globals_and_functions;
mod operators;
mod structs;
mod vector_construction;

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
