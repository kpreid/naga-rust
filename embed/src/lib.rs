//! Translates WGSL shader code to Rust embedded in your crate via macros.
//!
//! This does not necessarily mean you can run your compute or render pipelines in Rust
//! on your CPU unchanged; this is *not* a full “software renderer”. Rather, the primary goal
//! of the library is to allow you to share simple functions between CPU and GPU code, so
//! that the two parts of your code can agree on definitions.
//!
//! If you need additional control over the translation or to use a different source language,
//! use the [`naga_rust_back`] library directly instead.
//!
//! This library is in an early stage of development and many features do not work yet.
//! Expect compilation failures and to have to tweak your code to fit.
//! Broadly, simple mathematical functions will work, and bindings, textures, atomics,
//! derivatives, and workgroup operations will not.
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
/// If any configuration is needed, write it attribute-style before the source code literal:
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
/// If any configuration is needed, write it attribute-style before the source code literal:
///
/// ```
/// # use naga_rust_embed::wgsl;
/// wgsl!(
///     global_struct = Globals,
///     "var<private> foo: i32 = 10;",
/// );
///
/// assert_eq!(Globals::default().foo, naga_rust_embed::rt::Scalar(10));
/// ```
///
#[doc = include_str!("configuration_syntax.md")]
pub use naga_rust_macros::wgsl;

/// Support library for the generated Rust code.
/// Do not use this directly; its contents are not guaranteed to be stable.
pub use naga_rust_rt as rt;
