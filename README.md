Rust backend for Naga
=====================

These libraries allow you to translate shader code in any language supported by [Naga]
to Rust code.

This does not necessarily mean you can run your compute or render pipelines in Rust
on your CPU unchanged; this is *not* a full “software renderer”. Rather, the primary goal
of the project is to allow you to share selected *functions* between CPU and GPU, so
that they can agree on definitions that might be executed in either place.

[Naga]: https://crates.io/crates/naga

Packages in the workspace
-------------------------

* `naga-rust-back` is the Rust backend (code generator) itself.
* `naga-rust-rt` is the runtime support library, which provides data types and functions
  used by the generated code.
* `naga-rust-embed` provides macros for translating WGSL and embedding the Rust output
  in your Rust code.

Development status
------------------

These libraries are in an early stage of development and many features do not work yet.
Expect compilation failures, incorrect behaviors, and to have to tweak your code to fit,
if you wish to use them.
Broadly, simple mathematical functions will work, and bindings, textures, atomics,
derivatives, and workgroup operations will not.


License
-------

Copyright 2025 Kevin Reid and the gfx-rs authors.

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