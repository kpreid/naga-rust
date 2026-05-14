`naga-rust-embed`: embed WGSL code in your Rust code
====================================================

`naga-rust-embed` translates WGSL shader code to Rust embedded in your crate via macros.
You can use this to **share constants and functions between your GPU and CPU code**,
or to unit test your shader functions without setting up a pipeline or requiring a GPU device
in the tests.

You cannot use this to run your compute or render pipelines in Rust on your CPU unchanged;
this is not a full “software rendering” library and does not provide pipeline execution or 
triangle rasterization.

If you need additional control over the translation or to use a different source language,
use the [`naga-rust-back`] library directly instead.

This library is in an early stage of development and many features do not work yet.
Expect compilation failures, incorrect behaviors, and to have to tweak your code to fit,
if you wish to use them. Broadly:

* Simple mathematical functions will work.
* Code involving pointers is likely to fail to compile.
* Textures are supported but texture filtering is not.
* Storage buffers are not supported.
* Atomics, derivatives, and workgroup operations are not supported.
* Pipelines involving multiple shaders (e.g. passing data from vertex to fragment)
  are not automatically executed but you can, in principle, build that yourself.

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
