# Changelog for `naga-rust-back`

## Unreleased

This release focuses on expanding to a new use case: sharing declarations between Rust and shaders, without actually executing shader functions as Rust.
You can now ask for structs and constants, but not functions, to be translated, and derive traits on those Rust structs.

**Caveat:** We do not yet ensure that the layout of structs containing `vec3`s is correct.


## Added

* `Config::include_functions()`, if disabled, allows translating only `struct`s and `const`s.
* `Config::rule()` allows customizing the translation of specific parts of the shader code.
  The first available customization is to add `#[derive]` to structs.

## Changed

* The `naga` version is now 30.

## 0.2.0 (2026-05-14)

### Added

* Support for matrices.
* Support for texture loads (but not yet filtering).
* Support for `continuing` and `break if` control flow.
* Support for boolean vector functions `any()`, `all()`, and `select()`.
* Partial support for additional math functions, such as `cross()` and `smoothstep()`.
* Support for bit shifts.
* Structs defined in shader code now have their constructor functions (`::new()` in Rust).
* `Config::allow_unimplemented` allows ignoring unsupported features by panicking when they are used instead of refusing to translate.
* `Config::resource_struct` allows passing uniforms and textures to the shader.

### Changed

* The signatures of generated functions have been changed; whenever the shader code has a parameter `x: T` where `T` is some scalar or vector type, the generated code now uses `x: impl Into<T>`. (This allows passing arrays as arguments where vectors are wanted.)
* In some cases such as accessing constants, the type of scalars such as `f32` will be translated to `naga_rust_rt::Scalar<f32>` instead of Rust `f32`.

## 0.1.0 (2025-03-25)

Initial public release.
