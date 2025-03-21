use core::{cmp, ops};
use num_traits::ConstZero;

// -------------------------------------------------------------------------------------------------
// Vector type declarations.
//
// Note that these vectors are *not* prepared to become implemented as SIMD vectors.
// This is because, when SIMD happens, our SIMD story is going to be making the
// application-level vectors’ *components* into SIMD vectors, like:
//     Vec2<std::simd::Simd<f32, 4>>
// This will allow us to have a constant SIMD lane count across the entire execution, even while
// the shader code works with vectors of all sizes and scalars.

/// This type wraps an underlying Rust-native scalar (or someday SIMD) type
/// to provide *only* methods and operators which are compatible with shader
/// function behaviors. That is, they act like `Vec*` even where `T` might not.
/// This allows the code generator to not worry as much about mismatches
/// between Rust and shader semantics.
///
/// TODO: This isn't actually used yet.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Scalar<T> {
    pub x: T,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(C)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(C)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(C)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

// -------------------------------------------------------------------------------------------------
// Vector operations

/// Generate arithmetic operators and functions for cases where the element type is a
/// Rust primitive *integer*. (In these cases we must ask for wrapping operations.)
macro_rules! impl_vector_integer_arithmetic {
    ($vec:ident, $int:ty, $( $component:ident )*) => {
        // Vector-vector operations
        impl ops::Add for $vec<$int> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_add(rhs.$component), )* }
            }
        }
        impl ops::Sub for $vec<$int> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_sub(rhs.$component), )* }
            }
        }
        impl ops::Mul for $vec<$int> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_mul(rhs.$component), )* }
            }
        }
        impl ops::Div for $vec<$int> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self::Output {
                // wrapping_div() panics on division by zero, which is not what we need
                $vec { $(
                    $component:
                        self.$component.checked_div(rhs.$component)
                            .unwrap_or(self.$component),
                )* }
            }
        }
        impl ops::Rem for $vec<$int> {
            type Output = Self;
            fn rem(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_rem(rhs.$component), )* }
            }
        }

        // Vector-scalar operations
        impl ops::Add<$int> for $vec<$int> {
            type Output = Self;
            fn add(self, rhs: $int) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_add(rhs), )* }
            }
        }
        impl ops::Sub<$int> for $vec<$int> {
            type Output = Self;
            fn sub(self, rhs: $int) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_sub(rhs), )* }
            }
        }
        impl ops::Mul<$int> for $vec<$int> {
            type Output = Self;
            fn mul(self, rhs: $int) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_mul(rhs), )* }
            }
        }
        impl ops::Div<$int> for $vec<$int> {
            type Output = Self;
            fn div(self, rhs: $int) -> Self::Output {
                // wrapping_div() panics on division by zero, which is not what we need
                $vec { $(
                    $component:
                        self.$component.checked_div(rhs)
                            .unwrap_or(self.$component),
                )* }
            }
        }
        impl ops::Rem<$int> for $vec<$int> {
            type Output = Self;
            fn rem(self, rhs: $int) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_rem(rhs), )* }
            }
        }
    }
}

/// Generate arithmetic operators and functions for cases where the element type is a
/// Rust primitive *float*.
macro_rules! impl_vector_float_arithmetic {
    ($vec:ident, $int:ty, $( $component:ident )*) => {
        // Vector-vector operations
        impl ops::Add for $vec<$int> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component + rhs.$component, )* }
            }
        }
        impl ops::Sub for $vec<$int> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component - rhs.$component, )* }
            }
        }
        impl ops::Mul for $vec<$int> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component * rhs.$component, )* }
            }
        }
        impl ops::Div for $vec<$int> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component / rhs.$component, )* }
            }
        }
        impl ops::Rem for $vec<$int> {
            type Output = Self;
            fn rem(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component % rhs.$component, )* }
            }
        }

        // Vector-scalar operations
        impl ops::Add<$int> for $vec<$int> {
            type Output = Self;
            fn add(self, rhs: $int) -> Self::Output {
                $vec { $( $component: self.$component + rhs, )* }
            }
        }
        impl ops::Sub<$int> for $vec<$int> {
            type Output = Self;
            fn sub(self, rhs: $int) -> Self::Output {
                $vec { $( $component: self.$component - rhs, )* }
            }
        }
        impl ops::Mul<$int> for $vec<$int> {
            type Output = Self;
            fn mul(self, rhs: $int) -> Self::Output {
                $vec { $( $component: self.$component * rhs, )* }
            }
        }
        impl ops::Div<$int> for $vec<$int> {
            type Output = Self;
            fn div(self, rhs: $int) -> Self::Output {
                $vec { $( $component: self.$component / rhs, )* }
            }
        }
        impl ops::Rem<$int> for $vec<$int> {
            type Output = Self;
            fn rem(self, rhs: $int) -> Self::Output {
                $vec { $( $component: self.$component % rhs, )* }
            }
        }
    }
}

macro_rules! impl_element_casts {
    ($ty:ident) => {
        // TODO: These do not have the right cast semantics, but what *are* the right cast
        // semantics? Naga IR docs are cryptic for `Expression::As`.
        pub fn cast_elem_as_u32(self) -> $ty<u32> {
            self.map(|component| component as u32)
        }
        pub fn cast_elem_as_i32(self) -> $ty<i32> {
            self.map(|component| component as i32)
        }
        pub fn cast_elem_as_f32(self) -> $ty<f32> {
            self.map(|component| component as f32)
        }
        pub fn cast_elem_as_f64(self) -> $ty<f64> {
            self.map(|component| component as f64)
        }
    };
}

macro_rules! impl_vector_regular_fns {
    ( $ty:ident $component_count:literal : $( $component:ident )* ) => {
        impl<T> $ty<T> {
            pub const fn new($( $component: T, )*) -> Self {
                Self { $( $component, )* }
            }

            pub fn splat(value: T) -> Self
            where
                T: Copy
            {
                Self { $( $component: value, )* }
            }

            /// Replaces the elements of `self` with the elements of `trues` wherever
            /// `mask` contains [`true`].
            pub const fn select(self, trues: Self, mask: $ty<bool>) -> Self
            where
                T: Copy
            {
                Self {
                    $(
                        $component: if mask.$component {
                            trues.$component
                        } else {
                            self.$component
                        },
                    )*
                }
            }

            pub fn map<U, F>(self, mut f: F) -> $ty<U>
            where
                F: FnMut(T) -> U,
            {
                $ty {
                    $(
                        $component: f(self.$component),
                    )*
                }
            }
        }

        impl_vector_integer_arithmetic!($ty, i32, $($component)*);
        impl_vector_integer_arithmetic!($ty, u32, $($component)*);
        impl_vector_float_arithmetic!($ty, f32, $($component)*);
        impl_vector_float_arithmetic!($ty, f64, $($component)*);

        impl $ty<i32> {
            impl_element_casts!($ty);
        }
        impl $ty<u32> {
            impl_element_casts!($ty);
        }
        impl $ty<f32> {
            impl_element_casts!($ty);
        }
        impl $ty<f64> {
            impl_element_casts!($ty);
        }

        // Zero constant and traits.
        impl<T: ConstZero> $ty<T> {
            // inherent so that traits are not needed in scope
            pub const ZERO: Self = Self { $( $component: T::ZERO, )* };
        }
        impl<T: ConstZero> ConstZero for $ty<T>
        where
            Self: ops::Add<Output = Self>
        {
            const ZERO: Self = Self { $( $component: T::ZERO, )* };
        }
        impl<T: ConstZero> num_traits::Zero for $ty<T>
        where
            Self: ops::Add<Output = Self>
        {
            fn zero() -> Self {
                Self::ZERO
            }
            fn is_zero(&self) -> bool {
                $(T::is_zero(&self.$component) & )* true
            }
        }

        // Elementwise comparison
        impl<T: PartialOrd> $ty<T> {
            pub fn elementwise_eq(self, rhs: Self) -> $ty<bool> {
                self.partial_cmp(rhs).map(|cmp| matches!(cmp, Some(cmp::Ordering::Equal)))
            }
            pub fn elementwise_ne(self, rhs: Self) -> $ty<bool> {
                self.partial_cmp(rhs).map(|cmp| !matches!(cmp, Some(cmp::Ordering::Equal)))
            }
            pub fn elementwise_lt(self, rhs: Self) -> $ty<bool> {
                self.partial_cmp(rhs).map(|cmp| matches!(cmp, Some(cmp::Ordering::Less)))
            }
            pub fn elementwise_le(self, rhs: Self) -> $ty<bool> {
                self.partial_cmp(rhs).map(|cmp| {
                    matches!(cmp, Some(cmp::Ordering::Less | cmp::Ordering::Equal))
                })
            }
            pub fn elementwise_gt(self, rhs: Self) -> $ty<bool> {
                self.partial_cmp(rhs).map(|cmp| matches!(cmp, Some(cmp::Ordering::Greater)))
            }
            pub fn elementwise_ge(self, rhs: Self) -> $ty<bool> {
                self.partial_cmp(rhs).map(|cmp| {
                    matches!(cmp, Some(cmp::Ordering::Greater | cmp::Ordering::Equal))
                })
            }

            /// Helper for comparison operations
            fn partial_cmp(self, rhs: Self) -> $ty<Option<cmp::Ordering>> {
                $ty {
                    $(
                        $component: self.$component.partial_cmp(&rhs.$component),
                    )*
                }
            }
        }

        // Conversion in and out
        impl<T> From<[T; $component_count]> for $ty<T> {
            fn from(value: [T; $component_count]) -> Self {
                let [$( $component ),*] = value;
                Self::new($( $component ),*)
            }
        }
        impl<T> From<$ty<T>> for [T; $component_count] {
            fn from(value: $ty<T>) -> Self {
                [$( value.$component ),*]
            }
        }
    }
}

impl_vector_regular_fns!(Scalar 1 : x);
impl_vector_regular_fns!(Vec2 2 : x y);
impl_vector_regular_fns!(Vec3 3 : x y z);
impl_vector_regular_fns!(Vec4 4 : x y z w);

// -------------------------------------------------------------------------------------------------
// Irregular functions and impls

impl<T> From<T> for Scalar<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

// Constructor functions that take a mix of scalars and vectors.
// These names must match those generated by `Writer::write_constructor_expression()`.
macro_rules! impl_flattening_ctor {
    (fn $fn_name:ident ( $($param:tt)* ) => ( $($arg:tt)* )) => {
        pub const fn $fn_name ($($param)*) -> Self {
            Self::new($($arg)*)
        }
    }
}
// Note: Copy bound is solely due to otherwise needing `feature(const_precise_live_drops)`.
impl<T: Copy> Vec3<T> {
    impl_flattening_ctor!(fn new_12(x: T, yz: Vec2<T>) => (x, yz.x, yz.y));
    impl_flattening_ctor!(fn new_21(xy: Vec2<T>, z: T) => (xy.x, xy.y, z));
}
impl<T: Copy> Vec4<T> {
    impl_flattening_ctor!(fn new_112(x: T, y: T, zw: Vec2<T>) => (x, y, zw.x, zw.y));
    impl_flattening_ctor!(fn new_121(x: T, yz: Vec2<T>, w: T) => (x, yz.x, yz.y, w));
    impl_flattening_ctor!(fn new_211(xy: Vec2<T>, z: T, w: T) => (xy.x, xy.y, z, w));
    impl_flattening_ctor!(fn new_22(xy: Vec2<T>, zw: Vec2<T>) => (xy.x, xy.y, zw.x, zw.y));
    impl_flattening_ctor!(fn new_13(x: T, yzw: Vec3<T>) => (x, yzw.x, yzw.y, yzw.z));
    impl_flattening_ctor!(fn new_31(xyz: Vec3<T>, w: T) => (xyz.x, xyz.y, xyz.z, w));
}

// -------------------------------------------------------------------------------------------------
// Swizzles

macro_rules! swizzle_fn {
    ($name:ident $output:ident ($($cin:ident)*) ) => {
        /// Takes the
        #[doc = stringify!($(self.$cin )*)]
        /// elements of `self` and returns them in that order.
        pub fn $name(self) -> $output<T> {
            $output::new($(self.$cin,)*)
        }
    }
}

impl<T: Copy> Vec2<T> {
    swizzle_fn!(xy Vec2(x y));
    swizzle_fn!(yx Vec2(y x));
}
impl<T: Copy> Vec3<T> {
    swizzle_fn!(xy Vec2(x y));
    swizzle_fn!(yx Vec2(y x));
    swizzle_fn!(xyz Vec3(x y z));
    swizzle_fn!(xzy Vec3(x z y));
    swizzle_fn!(yxz Vec3(y x z));
    swizzle_fn!(yzx Vec3(y z x));
    swizzle_fn!(zxy Vec3(z x y));
    swizzle_fn!(zyx Vec3(z y x));
}
impl<T: Copy> Vec4<T> {
    swizzle_fn!(xy Vec2(x y));
    swizzle_fn!(yx Vec2(y x));
    swizzle_fn!(xyz Vec3(x y z));
    swizzle_fn!(xzy Vec3(x z y));
    swizzle_fn!(yxz Vec3(y x z));
    swizzle_fn!(yzx Vec3(y z x));
    swizzle_fn!(zxy Vec3(z x y));
    swizzle_fn!(zyx Vec3(z y x));
    swizzle_fn!(xyzw Vec4(x y z w));
    swizzle_fn!(xywz Vec4(x y w z));
    swizzle_fn!(xzwy Vec4(x z w y));
    swizzle_fn!(xzyw Vec4(x z y w));
    swizzle_fn!(xwyz Vec4(x w y z));
    swizzle_fn!(xwzy Vec4(x w z y));
    swizzle_fn!(yxzw Vec4(y x z w));
    swizzle_fn!(yxwz Vec4(y x w z));
    swizzle_fn!(yzxw Vec4(y z x w));
    swizzle_fn!(yzwx Vec4(y z w x));
    swizzle_fn!(ywzx Vec4(y w z x));
    swizzle_fn!(ywxz Vec4(y w x z));
    swizzle_fn!(zxyw Vec4(z x y w));
    swizzle_fn!(zxwy Vec4(z x w y));
    swizzle_fn!(zyxw Vec4(z y x w));
    swizzle_fn!(zywx Vec4(z y w x));
    swizzle_fn!(zwxy Vec4(z w x y));
    swizzle_fn!(zwyx Vec4(z w y x));
    swizzle_fn!(wxyz Vec4(w x y z));
    swizzle_fn!(wxzy Vec4(w x z y));
    swizzle_fn!(wyxz Vec4(w y x z));
    swizzle_fn!(wyzx Vec4(w y z x));
    swizzle_fn!(wzxy Vec4(w z x y));
    swizzle_fn!(wzyx Vec4(w z y x));
}
