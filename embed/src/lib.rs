//! Translates WGSL shader code to Rust embedded in your crate via macros.
//!
//! `naga-rust-embed` translates WGSL shader code to Rust embedded in your crate via macros.
//! You can use this to **share individual functions, constants, and data types between CPU and
//! GPU,** so that they can agree on definitions that might be executed in either place.
//! Potential applications include:
//!
//! * Unit tests of shader code, written as simple, cheap Rust `#[test]`s.
//! * Defining uniform buffer `struct`s in your shader, and then constructing them from Rust without
//!   needing to write a separate Rust `struct`.
//! * Sharing mathematical functions and constants which need to be used on both the CPU and GPU.
//! * Fallback CPU execution of code written for GPU execution, when no GPU is available.
//!
//! You cannot use this to run your compute or render pipelines in Rust on your CPU unchanged;
//! this is not a full “software renderer” and does not provide pipeline execution,
//! triangle rasterization, or even interpolation.
//!
//! This library is in an early stage of development and many features do not work yet.
//! Expect compilation failures, incorrect behaviors, and to have to tweak your code to fit,
//! if you wish to use it. Broadly:
//!
//! * Simple mathematical functions will work.
//! * Code involving pointers is likely to fail to compile.
//! * Textures are supported but texture filtering (use of samplers) is not.
//! * Storage buffers are not supported.
//! * Atomics, derivatives, and workgroup operations are not supported.
//! * Not only are whole pipelines not supported, there is no implementation of interpolation
//!   (as would occur when passing data from a vertex shader to a fragment shader).
//!
//! # Example
//!
// TODO: Make this example more obviously an example of WGSL and not Rust.
//! ```
//! naga_rust_embed::wgsl!(r"
//!     fn add_one(x: i32) -> i32 {
//!         return x + 1;
//!     }
//! ");
//!
//! assert_eq!(add_one(10), 11);
//! ```
//!
//! [`naga_rust_back`]: https://docs.rs/naga-rust-back
#![no_std]

/// Takes the pathname of a WGSL source file, as a string literal, and embeds its Rust translation.
///
/// The pathname must be relative to [`CARGO_MANIFEST_DIR`].
/// (If and when Rust proc-macros gain the ability to access files relative to the current
/// source file, a new `include_wgsl!` macro will be provided and this `include_wgsl_mr!` will be
/// deprecated.)
///
/// This macro should be used in a position where items are allowed
/// (e.g. inside a crate, module, function body, or block).
///
/// ```
/// # use naga_rust_embed::include_wgsl_mr;
/// include_wgsl_mr!("src/example.wgsl");
/// ```
///
/// If any configuration is needed, write it before the source code literal:
///
/// ```
/// # use naga_rust_embed::include_wgsl_mr;
/// include_wgsl_mr!(
///     global_struct = Globals,
///     "src/example.wgsl",
/// );
/// ```
///
#[doc = include_str!("configuration_syntax.md")]
///
/// [`CARGO_MANIFEST_DIR`]: https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates
pub use naga_rust_macros::include_wgsl_mr;

/// Converts the provided WGSL string literal to Rust.
///
/// The macro should be given a single string literal containing the source code,
/// and used in a position where items are allowed
/// (e.g. inside a crate, module, function body, or block).
///
/// ```
/// # use naga_rust_embed::wgsl;
/// wgsl!("fn wgsl_hello_world() {}");
///
/// fn main() {
///     wgsl_hello_world();
/// }
/// ```
///
/// If any configuration is needed, write it before the source code literal:
///
/// ```
/// # use naga_rust_embed::wgsl;
/// wgsl!(
///     global_struct = Globals,
///     "var<private> foo: i32 = 10;",
/// );
///
/// assert_eq!(Globals::new().foo, naga_rust_embed::rt::Scalar(10));
/// ```
///
#[doc = include_str!("configuration_syntax.md")]
pub use naga_rust_macros::wgsl;

/// Support library for the generated Rust code.
/// Do not use this directly; its contents are not guaranteed to be stable.
pub use naga_rust_rt as rt;
