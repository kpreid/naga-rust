`naga-rust-back`: translate shaders to Rust
===========================================

[`naga`] backend allowing you to translate shader (GPU) code in any language supported by Naga
to Rust code.

The generated code requires the [`naga-rust-rt`] library.
Alternatively, you can use [`naga-rust-embed`], which combines this library with [`naga-rust-rt`]
and provides convenient macros for embedding translated WGSL in your Rust code.

This library is in an early stage of development and many features do not work yet;
this may be indicated by returned errors or by the generated code failing to compile.
Broadly, simple mathematical functions will work, and bindings, textures, atomics,
derivatives, and workgroup operations will not.

[`naga`]: https://crates.io/crates/naga
[`naga-rust-rt`]: https://crates.io/crates/naga-rust-rt
[`naga-rust-embed`]: https://crates.io/crates/naga-rust-embed

License
-------

Copyright 2025 Kevin Reid and the gfx-rs authors.

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
