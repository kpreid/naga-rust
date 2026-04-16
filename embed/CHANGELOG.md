# Changelog for `naga-rust-embed`

## 0.2.0 (Unreleased)

### Added

* Support for matrices.
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
