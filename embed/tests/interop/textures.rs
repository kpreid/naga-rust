use core::cell::RefCell;
use core::marker::PhantomData;
use core::num::NonZeroU32;

use naga_rust_embed::rt::{self, Scalar, Vec2, Vec3, Vec4, texture::ONE};
use naga_rust_embed::wgsl;

// -------------------------------------------------------------------------------------------------

struct MockTextureStorage<C, F: ?Sized> {
    _phantom: PhantomData<fn(C)>,
    function: RefCell<F>,
}
impl<C, F> MockTextureStorage<C, F>
where
    C: Copy + 'static,
    F: FnMut(C, i32, i32, i32) -> Vec4<f32>,
{
    fn new(function: F) -> Self {
        Self {
            _phantom: PhantomData,
            function: RefCell::new(function),
        }
    }
}
impl<C, F> rt::texture::Read for MockTextureStorage<C, F>
where
    C: Copy + 'static,
    F: FnMut(C, i32, i32, i32) -> Vec4<f32>,
{
    type Coordinates = C;
    type Component = f32;

    fn read_texel(
        &self,
        coordinates: Self::Coordinates,
        array_layer: i32,
        sample: i32,
        mip_level: i32,
    ) -> Vec4<f32> {
        (self.function.borrow_mut())(coordinates, array_layer, sample, mip_level)
    }
}

// -------------------------------------------------------------------------------------------------

#[test]
fn query_and_load_1d() {
    wgsl!(
        resource_struct = Resources,
        r"
        @group(0) @binding(0) var my_texture: texture_1d<f32>;
        fn dimensions() -> u32 {
            return textureDimensions(my_texture, 0);
        }
        fn load(position: i32) -> vec4f {
            return textureLoad(my_texture, position, 0);
        }"
    );

    let mut call_count = 0;

    let ts = MockTextureStorage::new(|coordinates, _a, _s, mip_level| {
        assert_eq!(coordinates, Scalar::new(10));
        assert_eq!(mip_level, 0);
        call_count += 1;
        Vec4::new(1.0, 2.0, 3.0, 4.0)
    });

    let res = Resources {
        my_texture: rt::Texture1d {
            dimensions: Scalar::new(NonZeroU32::new(100).unwrap()),
            mip_levels: NonZeroU32::new(1).unwrap(),
            data: &ts,
        },
    };
    assert_eq!(res.dimensions(), 100);
    assert_eq!(res.load(Scalar::new(10)), Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(call_count, 1);
}

#[test]
fn query_and_load_2d() {
    wgsl!(
        resource_struct = Resources,
        r"
        @group(0) @binding(0) var my_texture: texture_2d<f32>;
        fn dimensions() -> vec2u {
            return textureDimensions(my_texture, 0);
        }
        fn load(position: vec2i) -> vec4f {
            return textureLoad(my_texture, position, 0);
        }"
    );

    let mut call_count = 0;

    let ts = MockTextureStorage::new(|coordinates, _a, _s, mip_level| {
        assert_eq!(coordinates, Vec2::new(10, 20));
        assert_eq!(mip_level, 0);
        call_count += 1;
        Vec4::new(1.0, 2.0, 3.0, 4.0)
    });

    let res = Resources {
        my_texture: rt::Texture2d {
            dimensions: Vec2::splat(NonZeroU32::new(100).unwrap()),
            mip_levels: NonZeroU32::new(1).unwrap(),
            data: &ts,
        },
    };
    assert_eq!(res.dimensions(), Vec2::new(100u32, 100));
    assert_eq!(res.load(Vec2::new(10, 20)), Vec4::new(1.0, 2.0, 3.0, 4.0));
    assert_eq!(call_count, 1);
}

#[test]
fn query_and_load_3d() {
    wgsl!(
        resource_struct = Resources,
        r"
        @group(0) @binding(0) var my_texture: texture_3d<f32>;
        fn dimensions() -> vec3u {
            return textureDimensions(my_texture, 0);
        }
        fn load(position: vec3i) -> vec4f {
            return textureLoad(my_texture, position, 0);
        }"
    );

    let mut call_count = 0;

    let ts = MockTextureStorage::new(|coordinates, _a, _s, mip_level| {
        assert_eq!(coordinates, Vec3::new(10, 20, 30));
        assert_eq!(mip_level, 0);
        call_count += 1;
        Vec4::new(1.0, 2.0, 3.0, 4.0)
    });

    let res = Resources {
        my_texture: rt::Texture3d {
            dimensions: Vec3::splat(NonZeroU32::new(100).unwrap()),
            mip_levels: NonZeroU32::new(1).unwrap(),
            data: &ts,
        },
    };
    assert_eq!(res.dimensions(), Vec3::new(100u32, 100, 100));
    assert_eq!(
        res.load(Vec3::new(10, 20, 30)),
        Vec4::new(1.0, 2.0, 3.0, 4.0)
    );
    assert_eq!(call_count, 1);
}

// TODO: test all texture types

// -------------------------------------------------------------------------------------------------

#[test]
fn mip_level_dimensions() {
    wgsl!(
        resource_struct = Resources,
        r"
        @group(0) @binding(0) var texture1d: texture_1d<f32>;
        @group(0) @binding(0) var texture2d: texture_2d<f32>;
        @group(0) @binding(0) var texture3d: texture_3d<f32>;
        struct Output {
            one: u32,
            two: vec2u,
            three: vec3u,
        }
        fn dimensions(mip_level: i32) -> Output {
            return Output(
                textureDimensions(texture1d, mip_level),
                textureDimensions(texture2d, mip_level),
                textureDimensions(texture3d, mip_level),
            );
        }
        "
    );

    let res = Resources {
        texture1d: rt::Texture1d {
            dimensions: Scalar::new(NonZeroU32::new(15).unwrap()),
            mip_levels: NonZeroU32::new(1).unwrap(),
            data: &rt::texture::Constant::new(Vec4::splat(0.0)),
        },
        texture2d: rt::Texture2d {
            dimensions: Vec2::new(ONE, NonZeroU32::new(15).unwrap()),
            mip_levels: NonZeroU32::new(1).unwrap(),
            data: &rt::texture::Constant::new(Vec4::splat(0.0)),
        },
        texture3d: rt::Texture3d {
            dimensions: Vec3::new(
                ONE,
                NonZeroU32::new(15).unwrap(),
                NonZeroU32::new(16).unwrap(),
            ),
            mip_levels: NonZeroU32::new(1).unwrap(),
            data: &rt::texture::Constant::new(Vec4::splat(0.0)),
        },
    };
    pretty_assertions::assert_eq!(
        [
            res.dimensions(0),
            res.dimensions(1),
            res.dimensions(2),
            res.dimensions(3),
            res.dimensions(4),
        ],
        [
            Output {
                one: 15,
                two: Vec2::new(1, 15),
                three: Vec3::new(1, 15, 16),
            },
            Output {
                one: 7,
                two: Vec2::new(1, 7),
                three: Vec3::new(1, 7, 8),
            },
            Output {
                one: 3,
                two: Vec2::new(1, 3),
                three: Vec3::new(1, 3, 4),
            },
            Output {
                one: 1,
                two: Vec2::new(1, 1),
                three: Vec3::new(1, 1, 2),
            },
            // out of range for all but the 3D example; indeterminate per spec
            // <<https://www.w3.org/TR/2026/CRD-WGSL-20260507/#texturedimensions>
            Output {
                one: 1,
                two: Vec2::new(1, 1),
                three: Vec3::new(1, 1, 1),
            },
        ]
    );
}

// -------------------------------------------------------------------------------------------------

/// We don’t implement texture sampling yet, but shaders that mentions samplers should still be
/// compilable.
#[test]
fn sampler() {
    wgsl!(
        resource_struct = Resources,
        "
        @group(0) @binding(0) var my_sampler: sampler;
        fn unrelated_function() {}
        "
    );

    // This should run without error (it does not use the sampler binding).
    Resources {
        my_sampler: rt::Sampler,
    }
    .unrelated_function();
}
