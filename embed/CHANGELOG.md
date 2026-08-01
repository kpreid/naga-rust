# Changelog for `naga-rust-embed`

## 0.3.1 (2026-07-31)

## Fixed

* `mix()` with 3 vector arguments no longer fails to compile.
* `textureLoad()` with `u32` coordinates now succeeds instead of generating a type error.
* The visibility of the `global_struct` and `resource_struct` is now controlled by the `public_items` configuration instead of always being private.

## 0.3.0 (2026-07-27)

This release focuses on expanding to a new use case: sharing declarations between Rust and shaders, without actually executing shader functions as Rust.
You can now ask for structs and constants, but not functions, to be translated, and derive traits on those Rust structs.

**Caveat:** We do not yet ensure that the layout of structs containing `vec3`s is correct.

### Added

* Configuration `include_functions`, if disabled, allows translating only `struct`s and `const`s.
* Configuration `rule`s allow customizing the translation of specific parts of the shader code.
  Currently available:
  * Adding `#[derive]` to selected structs.
  * Adding `#[inline]` to selected functions.
* Implementations of [`bytemuck` v1](https://docs.rs/bytemuck/1/bytemuck/)’s traits for our scalar, vector, and matrix types, with `features = ["bytemuck"]`.
  Note that this does not implement those traits on translated `struct` types; that may be done separately using the new `rule` feature.

### Changed

* The `naga` version is now 30.
* The macros’ parsing code has been rewritten.
  It no longer depends on the `syn` library, and has more specific error messages in several cases.

## 0.2.0 (2026-05-14)

### Added

* Support for matrices.
* Support for texture loads (but not yet filtering).
* Support for `continuing` and `break if` control flow.
* Support for boolean vector functions `any()`, `all()`, and `select()`.
* Partial support for additional math functions, such as `cross()` and `smoothstep()`.
* Support for bit shifts.
* Structs defined in shader code now have their constructor functions (`::new()` in Rust).
* Configuration `allow_unimplemented` allows ignoring unsupported features by panicking when they are used instead of refusing to translate.
* Configuration `resource_struct` allows passing uniforms and textures to the shader.

### Changed

* The signatures of generated functions have been changed; whenever the shader code has a parameter `x: T` where `T` is some scalar or vector type, the generated code now uses `x: impl Into<T>`. (This allows passing arrays as arguments where vectors are wanted.)
* In some cases such as accessing constants, the type of scalars such as `f32` will be translated to `naga_rust_rt::Scalar<f32>` instead of Rust `f32`.

## 0.1.0 (2025-03-25)

Initial public release.
