`naga-rust-embed`: embed WGSL code in your Rust code
====================================================

Translates WGSL shader code to Rust embedded in your crate via macros.

This does not necessarily mean you can run your compute or render pipelines in Rust
on your CPU unchanged; this is *not* a full “software renderer”. Rather, the primary goal
of the library is to allow you to share simple functions between CPU and GPU code, so
that the two parts of your code can agree on definitions.

If you need additional control over the translation or to use a different source language,
use the `naga_rust_back` library directly instead.

This library is in an early stage of development and many features do not work yet.
Expect compilation failures and to have to tweak your code to fit.

[`naga`]: https://crates.io/crates/naga

License
-------

Copyright 2025 Kevin Reid and the gfx-rs authors.

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
