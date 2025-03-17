//! Translates WGSL shader code to Rust you can embed in your Rust code.
//!
//! This does not necessarily mean you can run your compute or render pipelines in Rust
//! on your CPU unchanged; this is *not* a “software renderer”. Rather, the primary goal
//! of the project is to allow you to share simple functions between CPU and GPU code, so
//! that the two parts of your code can agree on definitions.
//!
//! If you need additional control over the translation or to use a different source language,
//! use the `naga_rust_back` library directly instead.
//!
//! # Example
//!
// TODO: Make this example more obviously an example of WGSL and not Rust.
//! ```
//! naga_rust::wgsl!(r"
//!     fn add_one(x: i32) -> i32 {
//!         return x + 1;
//!     }
//! ");
//!
//! assert_eq!(add_one(10), 11);
//! ```

#![no_std]

/// Takes the pathname of a WGSL source file, as a string literal, and embeds its Rust translation.
///
/// The pathname must be relative to [`CARGO_MANIFEST_DIR`].
/// (If and when Rust proc-macros gain the ability to access files relative to the current
/// source file, a new `include_wgsl!` macro will be provided and this `include_wgsl_mr!` will be
/// deprecated.)
///
/// TODO: Document the details of the translation.
///
/// [`CARGO_MANIFEST_DIR`]: https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates
pub use naga_rust_macros::include_wgsl_mr;

/// Converts the provided WGSL string literal to Rust.
///
/// TODO: Document the details of the translation.
pub use naga_rust_macros::wgsl;

/// Support library for the generated Rust code.
/// Do not use this directly; its contents are not guaranteed to be stable.
pub use naga_rust_rt as rt;
