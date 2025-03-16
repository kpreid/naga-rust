pub use glam::{
    BVec2, BVec3, BVec4, BVec4A, DMat2, DMat3, DMat4, DVec2, DVec3, DVec4, I8Vec2, I8Vec3, I8Vec4,
    I16Vec2, I16Vec3, I16Vec4, I64Vec2, I64Vec3, I64Vec4, IVec2, IVec3, IVec4, Mat2, Mat3, Mat4,
    U8Vec2, U8Vec3, U8Vec4, U16Vec2, U16Vec3, U16Vec4, U64Vec2, U64Vec3, U64Vec4, UVec2, UVec3,
    UVec4, Vec2, Vec3, Vec4,
};

// // TODO: These should be `mint` types or some other light vector library.
// pub type Vec2<T> = [T; 2];
// pub type Vec3<T> = [T; 3];
// pub type Vec4<T> = [T; 4];
//
// impl<T> crate::New for Vec2<T> {
//     type Args = (T, T);
//     fn new((x, y): Self::Args) -> Self {
//         Self::new(x, y)
//     }
// }
//
// impl<T> crate::New for Vec3<T> {
//     type Args = (T, T, T);
//     fn new((x, y, z): Self::Args) -> Self {
//         Self::new(x, y, z)
//     }
// }
//
// impl<T> crate::New for Vec4<T> {
//     type Args = (T, T, T, T);
//     fn new((x, y, z, w): Self::Args) -> Self {
//         Self::new(x, y, z, w)
//     }
// }

// pub fn vec2<T>(x: T, y: T) -> Vec2<T> {
//     [x, y]
// }
// pub fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
//     [x, y, z]
// }
// pub fn vec4<T>(x: T, y: T, z: T, w: T) -> Vec4<T> {
//     [x, y, z, w]
// }

/// Helper trait to pick a vector type given a scalar type and component count.
pub trait Splat<const N: usize> {
    type Vec;
    fn splat(self) -> Self::Vec;
}

#[inline(always)]
pub fn splat2<T: Splat<2>>(value: T) -> T::Vec {
    Splat::splat(value)
}
#[inline(always)]
pub fn splat3<T: Splat<3>>(value: T) -> T::Vec {
    Splat::splat(value)
}
#[inline(always)]
pub fn splat4<T: Splat<4>>(value: T) -> T::Vec {
    Splat::splat(value)
}

macro_rules! impl_splat {
    ($n:literal $scalar:ty => $vec:ty) => {
        impl Splat<$n> for $scalar {
            type Vec = $vec;
            #[inline(always)]
            fn splat(self) -> Self::Vec {
                // calls the inherent method, not this trait method
                <$vec>::splat(self)
            }
        }
    };
}
impl_splat!(2 f32 => Vec2);
impl_splat!(3 f32 => Vec3);
impl_splat!(4 f32 => Vec4);
impl_splat!(2 f64 => DVec2);
impl_splat!(3 f64 => DVec3);
impl_splat!(4 f64 => DVec4);
impl_splat!(2 i32 => IVec2);
impl_splat!(3 i32 => IVec3);
impl_splat!(4 i32 => IVec4);
impl_splat!(2 u32 => UVec2);
impl_splat!(3 u32 => UVec3);
impl_splat!(4 u32 => UVec4);
impl_splat!(2 i64 => I64Vec2);
impl_splat!(3 i64 => I64Vec3);
impl_splat!(4 i64 => I64Vec4);
impl_splat!(2 u64 => U64Vec2);
impl_splat!(3 u64 => U64Vec3);
impl_splat!(4 u64 => U64Vec4);
impl_splat!(2 i8 => I8Vec2);
impl_splat!(3 i8 => I8Vec3);
impl_splat!(4 i8 => I8Vec4);
impl_splat!(2 u8 => U8Vec2);
impl_splat!(3 u8 => U8Vec3);
impl_splat!(4 u8 => U8Vec4);
impl_splat!(2 i16 => I16Vec2);
impl_splat!(3 i16 => I16Vec3);
impl_splat!(4 i16 => I16Vec4);
impl_splat!(2 u16 => U16Vec2);
impl_splat!(3 u16 => U16Vec3);
impl_splat!(4 u16 => U16Vec4);
impl_splat!(2 bool => BVec2);
impl_splat!(3 bool => BVec3);
impl_splat!(4 bool => BVec4);

macro_rules! impl_cmp {
    ($name:ident 2 $val_vec:ident $op:tt) => {
        pub fn $name(a: $val_vec, b: $val_vec) -> BVec2 {
            BVec2::new(a.x $op b.x, a.y $op b.y)
        }
    };
    ($name:ident 3 $val_vec:ident $op:tt) => {
        pub fn $name(a: $val_vec, b: $val_vec) -> BVec3 {
            BVec3::new(a.x $op b.x, a.y $op b.y, a.z $op b.z)
        }
    };
    ($name:ident 4 $val_vec:ident $op:tt) => {
        pub fn $name(a: $val_vec, b: $val_vec) -> BVec4 {
            BVec4::new(a.x $op b.x, a.y $op b.y, a.z $op b.z, a.w $op b.w)
        }
    };
}

impl_cmp!(vec2_eq 2 Vec2 ==);
impl_cmp!(vec3_eq 3 Vec3 ==);
impl_cmp!(vec4_eq 4 Vec4 ==);
