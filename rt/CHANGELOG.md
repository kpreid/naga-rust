# Changelog for `naga-rust-rt`

## Unreleased

### Added

* `Vec*::from_array()`
* `Vec*::to_array()`
* `Mat*::from_column_arrays()`
* `Mat*::to_column_arrays()`
* Implementations of [`bytemuck` v1](https://docs.rs/bytemuck/1/bytemuck/)’s traits for our scalar, vector, and matrix types, with `features = ["bytemuck"]`.

## 0.2.0 (2026-05-14)

### Added

* Matrix types.
* `texture` module with `Texture` and `Sampler` structs, for providing textures to shaders.
* Many miscellaneous methods and trait implementations.

## 0.1.0 (2025-03-25)

Initial public release.
