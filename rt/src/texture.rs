use core::marker::PhantomData;
use core::num::NonZeroU32;

use crate::{Scalar, Vec2, Vec3, Vec4};

// -------------------------------------------------------------------------------------------------

/// Texture sampler (placeholder).
///
/// Use this type to satisfy a sampler binding in a resource struct.
pub struct Sampler;

/// The number 1, as a [`NonZeroU32`].
///
/// Convenience alias because textures heavily use non-zero sizes.
pub const ONE: NonZeroU32 = NonZeroU32::MIN;

// -------------------------------------------------------------------------------------------------

/// 1-dimensional texture object.
///
/// Use this type to satisfy a texture binding in a resource struct,
/// when the binding has WGSL type `texture_1d`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Texture1d<T: ?Sized> {
    pub dimensions: Scalar<NonZeroU32>,
    pub mip_levels: NonZeroU32,
    pub data: T,
}

/// 2-dimensional texture object.
///
/// Use this type to satisfy a texture binding in a resource struct,
/// when the binding has WGSL type `texture_2d` or `texture_external`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Texture2d<T: ?Sized> {
    pub dimensions: Vec2<NonZeroU32>,
    pub mip_levels: NonZeroU32,
    pub data: T,
}

/// 2-dimensional array texture object.
///
/// Use this type to satisfy a texture binding in a resource struct,
/// when the binding has WGSL type `texture_2d_array`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Texture2dArray<T: ?Sized> {
    pub dimensions: Vec2<NonZeroU32>,
    pub array_layers: NonZeroU32,
    pub mip_levels: NonZeroU32,
    pub data: T,
}

/// 3-dimensional texture object.
///
/// Use this type to satisfy a texture binding in a resource struct,
/// when the binding has WGSL type `texture_3d`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Texture3d<T: ?Sized> {
    pub dimensions: Vec3<NonZeroU32>,
    pub mip_levels: NonZeroU32,
    pub data: T,
}

/// Cubemap texture object.
///
/// Use this type to satisfy a texture binding in a resource struct,
/// when the binding has WGSL type `texture_cube`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TextureCube<T: ?Sized> {
    pub dimensions: Vec2<NonZeroU32>,
    pub mip_levels: NonZeroU32,
    pub data: T,
}

/// Cubemap array texture object.
///
/// Use this type to satisfy a texture binding in a resource struct,
/// when the binding has WGSL type `texture_cube_array`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TextureCubeArray<T: ?Sized> {
    pub dimensions: Vec2<NonZeroU32>,
    pub array_layers: NonZeroU32,
    pub mip_levels: NonZeroU32,
    pub data: T,
}

/// Multisampled texture object.
///
/// Use this type to satisfy a texture binding in a resource struct,
/// when the binding has WGSL type `texture_multisampled_2d`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TextureMultisampled2d<T: ?Sized> {
    pub dimensions: Vec2<NonZeroU32>,
    pub samples: NonZeroU32,
    pub data: T,
}

// -------------------------------------------------------------------------------------------------

// TODO: we want to have these convenient constructors but without a requirement for borrowing.
// We could solve that by using `Box` instead of `&`, but it would be better to work on a general
// mechanism to avoid `dyn`.

impl<'d, C: Component> Texture1d<&'d dyn Read<Coordinates = Scalar<i32>, Component = C>> {
    pub fn one_texel(texel: &'d Constant<Scalar<i32>, C>) -> Self {
        Self {
            dimensions: Scalar::new(ONE),
            mip_levels: ONE,
            data: texel,
        }
    }
}

impl<'d, C: Component> Texture2d<&'d dyn Read<Coordinates = Vec2<i32>, Component = C>> {
    pub fn one_texel(texel: &'d Constant<Vec2<i32>, C>) -> Self {
        Self {
            dimensions: Vec2::splat(ONE),
            mip_levels: ONE,
            data: texel,
        }
    }
}

// -------------------------------------------------------------------------------------------------

/// Reads one texel from a texture.
///
/// Implement this trait and then put the implementation in a [`Texture1d`], [`Texture2d`], etc.
/// to provide a texture to your shader code.
//
// TODO: Usage example.
pub trait Read {
    type Coordinates: Copy + 'static;
    type Component: Component;

    /// Loads a single texel from the texture.
    ///
    /// If the coordinates are out of bounds, the implementation should not panic,
    /// but perform one of the behaviors specified in <https://www.w3.org/TR/WGSL/#textureload>.
    fn read_texel(
        &self,
        coordinates: Self::Coordinates,
        array_layer: i32,
        sample: i32,
        mip_level: i32,
    ) -> Vec4<Self::Component>;
}

/// Information about a texture.
///
/// This trait is implemented by all the texture types like [`Texture1d`], [`Texture2d`], etc.
pub trait Query {
    /// Type of the dimensions of the texture.
    /// Should be a [`Scalar`], [`Vec2`], or [`Vec3`] of [`NonZeroU32`].
    type Dimensions: Dimensions;

    /// Numeric type exposed to the shader.
    ///
    /// Should be one of [`u32`], [`i32`], or [`f32`], and must match the parameter of the
    /// texture type in the shader.
    type Component: Component;

    /// Returns the dimensions of mip level 0 of the texture.
    fn base_dimensions(&self) -> Self::Dimensions;

    /// Returns the count of array layers of the texture.
    ///
    /// For non-array textures, returns 1.
    fn array_layers(&self) -> NonZeroU32 {
        ONE
    }

    /// Returns the count of mip levels of the texture.
    fn mip_levels(&self) -> NonZeroU32 {
        ONE
    }

    /// Returns the count of samples of the texture.
    ///
    /// For non-multisampled textures, returns 1.
    fn samples(&self) -> NonZeroU32 {
        ONE
    }
}

/// Types which can be components of texels.
pub trait Component: Copy + 'static {}
impl Component for u32 {}
impl Component for i32 {}
impl Component for f32 {}

// -------------------------------------------------------------------------------------------------

macro_rules! impl_read_forwarder_struct {
    ($texture_type:ident) => {
        impl<T: ?Sized + Read> Read for $texture_type<T> {
            type Coordinates = T::Coordinates;
            type Component = T::Component;

            fn read_texel(
                &self,
                coordinates: Self::Coordinates,
                array_layer: i32,
                sample: i32,
                mip_level: i32,
            ) -> Vec4<Self::Component> {
                self.data
                    .read_texel(coordinates, array_layer, sample, mip_level)
            }
        }
    };
}

impl_read_forwarder_struct!(Texture1d);
impl_read_forwarder_struct!(Texture2d);
impl_read_forwarder_struct!(Texture2dArray);
impl_read_forwarder_struct!(Texture3d);
impl_read_forwarder_struct!(TextureCube);
impl_read_forwarder_struct!(TextureCubeArray);
impl_read_forwarder_struct!(TextureMultisampled2d);

macro_rules! impl_read_forwarder_deref {
    ($($ty:tt)*) => {
        impl<T: ?Sized + Read> Read for $($ty)* {
            type Coordinates = T::Coordinates;
            type Component = T::Component;

            fn read_texel(
                &self,
                coordinates: Self::Coordinates,
                array_layer: i32,
                sample: i32,
                mip_level: i32,
            ) -> Vec4<Self::Component> {
                T::read_texel(&**self, coordinates, array_layer, sample, mip_level)
            }
        }
    };
}

impl_read_forwarder_deref!(&T);
impl_read_forwarder_deref!(&mut T);
#[cfg(feature = "alloc")]
impl_read_forwarder_deref!(alloc::boxed::Box<T>);
#[cfg(feature = "alloc")]
impl_read_forwarder_deref!(alloc::rc::Rc<T>);
#[cfg(feature = "alloc")]
impl_read_forwarder_deref!(alloc::sync::Arc<T>);

// -------------------------------------------------------------------------------------------------

macro_rules! query_common {
    () => {
        type Component = T::Component;
        fn base_dimensions(&self) -> Self::Dimensions {
            self.dimensions
        }
    };
}

macro_rules! query_mip {
    () => {
        fn mip_levels(&self) -> NonZeroU32 {
            self.mip_levels
        }
    };
}

impl<T: ?Sized + Read> Query for Texture1d<T> {
    type Dimensions = Scalar<NonZeroU32>;
    query_common!();
    query_mip!();
}

impl<T: ?Sized + Read> Query for Texture2d<T> {
    type Dimensions = Vec2<NonZeroU32>;
    query_common!();
    query_mip!();
}

impl<T: ?Sized + Read> Query for Texture2dArray<T> {
    type Dimensions = Vec2<NonZeroU32>;
    query_common!();
    query_mip!();
    fn array_layers(&self) -> NonZeroU32 {
        self.array_layers
    }
}

impl<T: ?Sized + Read> Query for Texture3d<T> {
    type Dimensions = Vec3<NonZeroU32>;
    query_common!();
    query_mip!();
}

impl<T: ?Sized + Read> Query for TextureCube<T> {
    type Dimensions = Vec2<NonZeroU32>;
    query_common!();
    query_mip!();
}

impl<T: ?Sized + Read> Query for TextureCubeArray<T> {
    type Dimensions = Vec2<NonZeroU32>;
    query_common!();
    query_mip!();
    fn array_layers(&self) -> NonZeroU32 {
        self.array_layers
    }
}

impl<T: ?Sized + Read> Query for TextureMultisampled2d<T> {
    type Dimensions = Vec2<NonZeroU32>;
    query_common!();
    fn samples(&self) -> NonZeroU32 {
        self.samples
    }
}

// -------------------------------------------------------------------------------------------------

/// Generic texture data container for textures whose texels are all identical.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)] // TODO: derives have bounds on C
pub struct Constant<C, T> {
    pub texel: Vec4<T>,
    _phantom: PhantomData<fn(C)>,
}

impl<C: Copy + 'static, T: Component> Constant<C, T> {
    pub const fn new(texel: Vec4<T>) -> Self {
        Self {
            texel,
            _phantom: PhantomData,
        }
    }
}

impl<C: Copy + 'static, T: Component> Read for Constant<C, T> {
    type Coordinates = C;
    type Component = T;

    fn read_texel(
        &self,
        _coordinates: Self::Coordinates,
        _array_layer: i32,
        _sample: i32,
        _mip_level: i32,
    ) -> Vec4<Self::Component> {
        self.texel
    }
}

// -------------------------------------------------------------------------------------------------

/// Computes the dimensions of a specific mip level of a texture.
///
/// The result is meaningless if the mip level is out of bounds.
//
// Design note: The result is `u32`, not `NonZeroU32`, for the sake of the shader code which does
// not have `NonZero` types.
pub fn dimensions<Q>(texture: &Q, mip_level: i32) -> <Q::Dimensions as Dimensions>::Plain
where
    Q: ?Sized + Query,
{
    Dimensions::at_mip_level(texture.base_dimensions(), mip_level as u32)
}

use dim::Dimensions;
mod dim {
    use super::*;

    /// Semi-private trait implemented for vectors that can describe texture dimensions.
    /// Helper for [`dimensions()`].
    pub trait Dimensions: Copy + 'static {
        /// The vector with `u32` components instead of `NonZeroU32`.
        type Plain;

        fn at_mip_level(self, mip_level: u32) -> Self::Plain;
    }

    impl Dimensions for Scalar<NonZeroU32> {
        type Plain = Scalar<u32>;
        fn at_mip_level(self, mip_level: u32) -> Self::Plain {
            Scalar::new(mip_divide_size(self.0, mip_level))
        }
    }
    impl Dimensions for Vec2<NonZeroU32> {
        type Plain = Vec2<u32>;
        fn at_mip_level(self, mip_level: u32) -> Self::Plain {
            Vec2::new(
                mip_divide_size(self.x, mip_level),
                mip_divide_size(self.y, mip_level),
            )
        }
    }
    impl Dimensions for Vec3<NonZeroU32> {
        type Plain = Vec3<u32>;
        fn at_mip_level(self, mip_level: u32) -> Self::Plain {
            Vec3::new(
                mip_divide_size(self.x, mip_level),
                mip_divide_size(self.y, mip_level),
                mip_divide_size(self.z, mip_level),
            )
        }
    }

    /// Computes one component of the size of a given mip level of a texture,
    /// given the size at level 0.
    fn mip_divide_size(size: NonZeroU32, mip_level: u32) -> u32 {
        // Per <https://www.w3.org/TR/2026/CRD-webgpu-20260507/#abstract-opdef-logical-miplevel-specific-texture-extent>
        (size.get() >> mip_level).max(1)
    }
}
