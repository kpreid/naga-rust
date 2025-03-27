# Changelog for `naga-rust-back`

## 0.2.0 (Unreleased)

* The signatures of generated functions have been changed; whenever the shader code has a parameter `x: T` where `T` is some scalar or vector type, the generated code now uses `x: impl Into<T>`. (This allows passing arrays as arguments where vectors are wanted.)

## 0.1.0 (2025-03-25)

Initial public release.
