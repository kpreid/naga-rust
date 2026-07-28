`naga-rust-embed`: embed WGSL code in your Rust code
====================================================

`naga-rust-embed` translates WGSL shader code to Rust embedded in your crate via macros.
You can use this to **share individual functions, constants, and data types between CPU and GPU,**
so that they can agree on definitions that might be executed in either place.
Potential applications include:

* Unit tests of shader code, written as simple, cheap Rust `#[test]`s.
* Defining uniform buffer `struct`s in your shader, and then constructing them from Rust without
  needing to write a separate Rust `struct`.
* Sharing mathematical functions and constants which need to be used on both the CPU and GPU.
* Fallback CPU execution of code written for GPU execution, when no GPU is available.

You cannot use this to run your compute or render pipelines in Rust on your CPU unchanged;
this is not a full “software renderer” and does not provide pipeline execution,
triangle rasterization, or even interpolation.

If you need additional control over the translation or to use a different source language,
use the [`naga-rust-back`] library directly instead.

Development status
------------------

This library is in an early stage of development and many features do not work yet.
Expect compilation failures, incorrect behaviors, and to have to tweak your code to fit,
if you wish to use them. Broadly:

* Simple mathematical functions will work.
* Code involving pointers is likely to fail to compile.
* Textures are supported but texture filtering (use of samplers) is not.
* Storage buffers are not supported.
* Atomics, derivatives, and workgroup operations are not supported.
* Not only are whole pipelines not supported, there is no implementation of interpolation
  (as would occur when passing data from a vertex shader to a fragment shader).

[`naga`]: https://crates.io/crates/naga
[`naga-rust-back`]: https://crates.io/crates/naga-rust-back

License
-------

Copyright 2025-2026 Kevin Reid and the gfx-rs authors.

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
