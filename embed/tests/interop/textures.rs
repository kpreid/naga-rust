use core::num::NonZeroU32;

use naga_rust_embed::rt::{self, Vec2, Vec4};
use naga_rust_embed::wgsl;

#[test]
fn texture_query_and_load() {
    struct MyTexture;
    impl rt::Texture for MyTexture {
        type Dimensions = Vec2<u32>;
        type Coordinates = Vec2<i32>;
        type Scalar = f32;

        fn dimensions(&self, _mip_level: i32) -> Self::Dimensions {
            Vec2::new(100, 100)
        }

        fn array_layers(&self) -> NonZeroU32 {
            NonZeroU32::MIN
        }

        fn mip_levels(&self) -> NonZeroU32 {
            NonZeroU32::MIN
        }

        fn samples(&self) -> NonZeroU32 {
            NonZeroU32::MIN
        }

        fn load(
            &self,
            coordinates: Self::Coordinates,
            _array_layer: i32,
            _sample: i32,
            mip_level: i32,
        ) -> Vec4<f32> {
            assert_eq!(coordinates, Vec2::new(10, 20));
            assert_eq!(mip_level, 0);
            Vec4::new(1.0, 2.0, 3.0, 4.0)
        }
    }

    // TODO: for full coverage, we need many different texture binding types and their corresponding
    // call overloads.
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

    let res = Resources {
        my_texture: &MyTexture,
    };
    assert_eq!(res.dimensions(), Vec2::new(100u32, 100));
    assert_eq!(res.load(Vec2::new(10, 20)), Vec4::new(1.0, 2.0, 3.0, 4.0));
}

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
