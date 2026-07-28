Rust backend for Naga
=====================

These libraries allow you to translate shader code in any language supported by [Naga]
to Rust code.

This is *not* a “software renderer” and cannot run your compute or render pipelines unchanged.
Rather, the primary goal of the project is to allow you to **share individual functions, constants,
and data types between CPU and GPU,**
so that they can agree on definitions that might be executed in either place.
Potential applications include:

* Unit tests of shader code, written as simple, cheap Rust `#[test]`s.
* Defining uniform buffer `struct`s in your shader, and then constructing them from Rust without
  needing to write a separate Rust `struct`.
* Sharing mathematical functions and constants which need to be used on both the CPU and GPU.
* Fallback CPU execution of code written for GPU execution, when no GPU is available.

[Naga]: https://crates.io/crates/naga

Packages in this workspace
--------------------------

* [`naga-rust-embed`](embed/) ([crates.io](https://crates.io/crates/naga-rust-embed),
  [docs.rs](https://docs.rs/naga-rust-embed/latest/))
  provides macros for translating WGSL and embedding the Rust output in your Rust code.
  In most cases, this is the only library package you need.
* [`naga-rust-back`](back/) ([crates.io](https://crates.io/crates/naga-rust-back),
  [docs.rs](https://docs.rs/naga-rust-back/latest/))
  is the Rust backend (code generator) itself.
  Use this to generate Rust code from a build script, or to use an input language other than WGSL.
* [`naga-rust-rt`](rt/) ([crates.io](https://crates.io/crates/naga-rust-rt),
  [docs.rs](https://docs.rs/naga-rust-rt/latest/))
  is the runtime support library, which provides data types and functions used by
  the generated code.

Development status
------------------

These libraries are in an early stage of development and many features do not work yet.
Expect compilation failures, incorrect behaviors, and to have to tweak your code to fit,
if you wish to use them. Broadly:

* Simple mathematical functions will work.
* Code involving pointers is likely to fail to compile.
* Textures are supported but texture filtering (use of samplers) is not.
* Storage buffers are not supported.
* Atomics, derivatives, and workgroup operations are not supported.
* Not only are whole pipelines not supported, there is no implementation of interpolation
  (as would occur when passing data from a vertex shader to a fragment shader).

### Isn’t there something that already does these things?

Yes, and some of them are much more powerful and feature-complete,
but here’s why `naga-rust` is interesting anyway:

* [Rust GPU](https://rust-gpu.github.io/) lets you write shared code in Rust instead of a shader
  language, but it requires you to use a pinned nightly toolchain version and uses a custom `rustc`
  backend, whereas `naga-rust` can be used with stable toolchains and runs as a proc-macro
  (or build script, if you prefer).
* [`shadybug`](https://docs.rs/shadybug/) is a complete software renderer that lets you write its
  shaders in Rust, but it does not provide GPU execution of the same code.

<!--
  `naga-rust` doesn’t yet handle shader `vec3` layout, but when it does, we can say that:

  * [`crevice`](https://docs.rs/crevice/), [`encase`](https://docs.rs/encase/), and
    [`const_shader_layout`](https://docs.rs/const_shader_layout/) help you write
    shader-compatible struct layouts, but don’t read your shader code to do it for you.
-->

License
-------

Copyright 2025-2026 Kevin Reid and the gfx-rs authors.

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

Contribution
------------

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.