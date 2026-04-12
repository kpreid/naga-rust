use core::num::NonZeroU32;

use crate::Vec4;

/// Texture sampler (placeholder).
///
/// Use this type to satisfy a sampler binding in a resource struct.
pub struct Sampler;

// TODO: Consider, instead of putting all the metadata as methods on the Txture trait,
// defining a struct with the metadata. This ensures consistency (particularly for things like
// per-mip-level dimensions) and avoids ever using dynamic dispatch, at the price of possibly
// duplicating some integers.

/// Implement this trait to provide a texture to the shader code.
// TODO: This will need expansion to handle depth textures
pub trait Texture {
    /// Type of the dimensions of the texture.
    /// Should be a [`Scalar`][crate::Scalar], [`Vec2`][crate::Vec2], or [`Vec3`][crate::Vec3]
    /// of `u32`.
    type Dimensions: Copy + 'static;

    /// Type of a point within the texture.
    /// Should be a [`Scalar`][crate::Scalar], [`Vec2`][crate::Vec2], or [`Vec3`][crate::Vec3]
    /// of `i32`.
    type Coordinates: Copy + 'static;

    /// Returns the dimensions of the texture.
    fn dimensions(&self, mip_level: i32) -> Self::Dimensions;

    /// Returns the count of array layers of the texture.
    fn array_layers(&self) -> NonZeroU32;

    /// Returns the count of mip levels of the texture.
    fn mip_levels(&self) -> NonZeroU32;

    /// Returns the count of samples of the texture.
    fn samples(&self) -> NonZeroU32;

    /// Loads a single texel from the texture.
    ///
    /// If the coordinates are out of bounds, do not panic, but perform one of the behaviors
    /// specified in <https://www.w3.org/TR/WGSL/#textureload>.
    fn load(
        &self,
        coordinates: Self::Coordinates,
        array_layer: i32,
        sample: i32,
        mip_level: i32,
    ) -> Vec4<f32>;
}
