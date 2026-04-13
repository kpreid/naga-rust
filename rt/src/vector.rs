use core::{cmp, fmt, ops};
use num_traits::ConstZero;

// Provides float math functions without std.
// TODO: Be more rigorous and explicitly call libm depending on the feature flag
// instead of using the trait.
#[cfg(not(feature = "std"))]
#[cfg_attr(test, allow(unused_imports))]
use num_traits::float::Float as _;

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
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Scalar<T>(pub T);

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(C)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(C)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(C)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

// -------------------------------------------------------------------------------------------------
// Most general helper macros

macro_rules! delegate_unary_method_elementwise {
    (const $name:ident ($($component:tt)*)) => {
        #[inline]
        pub const fn $name(self) -> Self {
            Self { $( $component: self.$component.$name() ),* }
        }
    };
    ($name:ident ($($component:tt)*)) => {
        #[inline]
        pub fn $name(self) -> Self {
            Self { $( $component: self.$component.$name() ),* }
        }
    };
}

macro_rules! delegate_unary_methods_elementwise {
    (const { $($name:ident),* } $components:tt) => {
        $( delegate_unary_method_elementwise!(const $name $components ); )*
    };
    ({ $($name:ident),* } $components:tt) => {
        $( delegate_unary_method_elementwise!($name $components ); )*
    };
}

macro_rules! delegate_binary_method_elementwise {
    (const $name:ident ($($component:tt)*)) => {
        #[inline]
        pub const fn $name(self, rhs: Self) -> Self {
            Self { $( $component: self.$component.$name(rhs.$component) ),* }
        }
    };
    ($name:ident ($($component:tt)*)) => {
        #[inline]
        pub fn $name(self, rhs: Self) -> Self {
            Self { $( $component: self.$component.$name(rhs.$component) ),* }
        }
    };
}

macro_rules! delegate_binary_methods_elementwise {
    (const { $($name:ident),* } $components:tt) => {
        $( delegate_binary_method_elementwise!(const $name $components ); )*
    };
    ({ $($name:ident),* } $components:tt) => {
        $( delegate_binary_method_elementwise!($name $components ); )*
    };
}

// -------------------------------------------------------------------------------------------------
// Vector operations

/// Generate arithmetic operators and functions for cases where the element type is a
/// Rust primitive *integer*. (In these cases we must ask for wrapping operations.)
///
/// Note that this macro provides binary operation impls for `vec op vec` and `scalar op scalar`
/// (considering a scalar as a 1-element vector) but not `scalar op vec` or `vec op scalar`.
macro_rules! impl_vector_integer_arithmetic {
    ($vec:ident, $int:ty, $( $component:tt )*) => {
        impl ops::Add for $vec<$int> {
            type Output = Self;
            /// Wrapping addition.
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_add(rhs.$component), )* }
            }
        }
        impl ops::Sub for $vec<$int> {
            type Output = Self;
            /// Wrapping subtraction.
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_sub(rhs.$component), )* }
            }
        }
        impl ops::Neg for $vec<$int> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_neg(), )* }
            }
        }
        impl ops::Mul for $vec<$int> {
            type Output = Self;
            /// Wrapping multiplication.
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_mul(rhs.$component), )* }
            }
        }
        impl ops::Div for $vec<$int> {
            type Output = Self;
            /// On division by zero or overflow, returns the component of `self`,
            /// per [WGSL](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#arithmetic-expr).
            #[inline]
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
            #[inline]
            fn rem(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_rem(rhs.$component), )* }
            }
        }

        impl $vec<$int> {
            /// As per WGSL [`clamp()`](https://www.w3.org/TR/2026/CRD-WGSL-20260310/#clamp).
            #[inline]
            pub fn clamp(self, low: Self, high: Self) -> Self {
                // std clamp() panics if low > high, which isn’t conformant
                $vec { $( $component: self.$component.max(low.$component).min(high.$component) ),*  }
            }

            delegate_binary_methods_elementwise!({
                max, min
            } ($($component)*));
        }
    }
}

/// As impl_vector_integer_arithmetic! but for vector-scalar ops (not vector-vector or
/// scalar-scalar).
macro_rules! impl_vector_scalar_integer_arithmetic {
    ($vec:ident, $int:ty, $( $component:tt )*) => {
        impl ops::Add<Scalar<$int>> for $vec<$int> {
            type Output = Self;
            /// Wrapping addition.
            #[inline]
            fn add(self, Scalar(rhs): Scalar<$int>) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_add(rhs), )* }
            }
        }
        impl ops::Sub<Scalar<$int>> for $vec<$int> {
            type Output = Self;
            /// Wrapping subtraction.
            #[inline]
            fn sub(self, Scalar(rhs): Scalar<$int>) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_sub(rhs), )* }
            }
        }
        // Subtraction is non-commutative so we need a dedicated scalar-vector impl
        impl ops::Sub<$vec<$int>> for Scalar<$int> {
            type Output = $vec<$int>;
            #[inline]
            fn sub(self, rhs: $vec<$int>) -> Self::Output {
                $vec { $( $component: self.0.wrapping_sub(rhs.$component), )* }
            }
        }
        impl ops::Mul<Scalar<$int>> for $vec<$int> {
            type Output = Self;
            /// Wrapping multiplication.
            #[inline]
            fn mul(self, Scalar(rhs): Scalar<$int>) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_mul(rhs), )* }
            }
        }
        impl ops::Div<Scalar<$int>> for $vec<$int> {
            type Output = Self;
            #[inline]
            /// On division by zero or overflow, returns `self`,
            /// per [WGSL](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#arithmetic-expr).
            fn div(self, Scalar(rhs): Scalar<$int>) -> Self::Output {
                // wrapping_div() panics on division by zero, which is not what we need
                $vec { $(
                    $component:
                        self.$component.checked_div(rhs)
                            .unwrap_or(self.$component),
                )* }
            }
        }
        impl ops::Div<$vec<$int>> for Scalar<$int> {
            type Output = $vec<$int>;
            #[inline]
            fn div(self, rhs: $vec<$int>) -> Self::Output {
                // wrapping_div() panics on division by zero, which is not what we need
                $vec { $(
                    $component:
                        self.0.checked_div(rhs.$component)
                            .unwrap_or(self.0),
                )* }
            }
        }
        impl ops::Rem<Scalar<$int>> for $vec<$int> {
            type Output = Self;
            #[inline]
            fn rem(self, Scalar(rhs): Scalar<$int>) -> Self::Output {
                $vec { $( $component: self.$component.wrapping_rem(rhs), )* }
            }
        }
    }
}

/// Generate arithmetic operators and functions for cases where the element type is a
/// Rust primitive *float*.
macro_rules! impl_vector_float_arithmetic {
    ($vec:ident, $float:ty, $( $component:tt )*) => {
        // Vector-vector operations
        impl ops::Add for $vec<$float> {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component + rhs.$component, )* }
            }
        }
        impl ops::Sub for $vec<$float> {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component - rhs.$component, )* }
            }
        }
        impl ops::Neg for $vec<$float> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                $vec { $( $component: -self.$component, )* }
            }
        }
        impl ops::Mul for $vec<$float> {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component * rhs.$component, )* }
            }
        }
        impl ops::Div for $vec<$float> {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component / rhs.$component, )* }
            }
        }
        impl ops::Rem for $vec<$float> {
            type Output = Self;
            #[inline]
            fn rem(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component % rhs.$component, )* }
            }
        }

        // Float math functions (mostly elementwise, but not exclusively)
        impl $vec<$float> {
            /// As per WGSL [`clamp()`](https://www.w3.org/TR/2026/CRD-WGSL-20260310/#clamp).
            #[inline]
            pub const fn clamp(self, low: Self, high: Self) -> Self {
                // std clamp() panics if low > high, which isn’t conformant
                $vec { $( $component: self.$component.max(low.$component).min(high.$component) ),*  }
            }
            /// As per WGSL [`distance()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#distance-builtin).
            #[inline]
            pub fn distance(self, rhs: Self) -> Scalar<$float> {
                (self - rhs).length()
            }
            /// As per WGSL [`faceForward()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#faceForward-builtin).
            #[inline]
            pub fn face_forward(self, e2: Self, e3: Self) -> Self {
                // note this is Rust's definition of signum which has no zero
                self * Scalar((-e2.dot(e3)).0.signum())
            }
            /// As per WGSL [`length()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#length-builtin).
            #[inline]
            pub fn length(self) -> Scalar<$float> {
                Scalar(self.dot(self).0.sqrt())
            }
            /// As per WGSL [`mix()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#mix-builtin).
            #[inline]
            pub const fn mix(self, rhs: Self, blend: Scalar<$float>) -> Self {
                $vec { $( $component: self.$component * (1.0 - blend.0) + rhs.$component * blend.0 ),*  }
            }
            /// As per WGSL [`fma()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#fma-builtin).
            #[inline]
            pub fn mul_add(self, b: Self, c: Self) -> Self {
                $vec { $( $component: self.$component.mul_add(b.$component, c.$component) ),*  }
            }
            /// As per WGSL [`normalize()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#normalize-builtin).
            #[inline]
            pub fn normalize(self) -> Self {
                self / self.length()
            }
            /// As per WGSL [`reflect()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#reflect-builtin).
            #[inline]
            pub fn reflect(self, rhs: Self) -> Self {
                self - rhs * (Scalar(2.0) * rhs.dot(self))
            }
            /// As per WGSL [`saturate()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#saturate-float-builtin).
            #[inline]
            pub const fn saturate(self) -> Self {
                $vec { $( $component: self.$component.clamp(0.0, 1.0) ),* }
            }
            /// As per WGSL [`sign()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#sign-builtin).
            #[inline]
            pub const fn sign(self) -> Self {
                $vec { $(
                    // TODO: branchless form of this?
                    $component: if self.$component == 0.0 {
                        0.0
                    } else {
                        self.$component.signum()
                    }
                ),* }
            }
            /// As per WGSL [`smoothstep()`](https://www.w3.org/TR/2026/CRD-WGSL-20260310/#smoothstep-builtin),
            /// but the third parameter called `x` is moved to be the `self` parameter.
            pub fn smoothstep(self, edge0: Self, edge1: Self) -> Self {
                let t = ((self - edge0) / (edge1 - edge0)).saturate();
                t * t * (Scalar(3.0) - Scalar(2.0) * t)
            }

            // TODO: some more of these can be const
            delegate_unary_methods_elementwise!(const {
                abs
            } ($($component)*));
            delegate_unary_methods_elementwise!({
                acos, acosh, asin, asinh, atan, atanh, ceil, cos, cosh, exp, exp2, floor, fract, log2, round,
                sin, sinh, tan, tanh, trunc, to_degrees, to_radians
            } ($($component)*));
            delegate_binary_methods_elementwise!({
                atan2, max, min, powf
            } ($($component)*));

            // TODO: modf, frexp, ldexp, cross, refract, step, smoothstep, inverse_sqrt,
            // quantizeToF16, pack*,
        }
    }
}

/// As impl_vector_float_arithmetic! but for vector-scalar ops (not vector-vector or
/// scalar-scalar).
macro_rules! impl_vector_scalar_float_arithmetic {
    ($vec:ident, $float:ty, $( $component:tt )*) => {
        impl ops::Add<Scalar<$float>> for $vec<$float> {
            type Output = $vec<$float>;
            #[inline]
            fn add(self, rhs: Scalar<$float>) -> Self::Output {
                $vec { $( $component: self.$component + rhs.0, )* }
            }
        }
        impl ops::Sub<Scalar<$float>> for $vec<$float> {
            type Output = $vec<$float>;
            #[inline]
            fn sub(self, rhs: Scalar<$float>) -> Self::Output {
                $vec { $( $component: self.$component - rhs.0, )* }
            }
        }
        impl ops::Sub<$vec<$float>> for Scalar<$float> {
            type Output = $vec<$float>;
            #[inline]
            fn sub(self, rhs: $vec<$float>) -> Self::Output {
                $vec { $( $component: self.0 - rhs.$component, )* }
            }
        }
        impl ops::Mul<Scalar<$float>> for $vec<$float> {
            type Output = $vec<$float>;
            #[inline]
            fn mul(self, rhs: Scalar<$float>) -> Self::Output {
                $vec { $( $component: self.$component * rhs.0, )* }
            }
        }
        impl ops::Div<Scalar<$float>> for $vec<$float> {
            type Output = $vec<$float>;
            #[inline]
            fn div(self, rhs: Scalar<$float>) -> Self::Output {
                $vec { $( $component: self.$component / rhs.0, )* }
            }
        }
        impl ops::Div<$vec<$float>> for Scalar<$float> {
            type Output = $vec<$float>;
            #[inline]
            fn div(self, rhs: $vec<$float>) -> Self::Output {
                $vec { $( $component: self.0 / rhs.$component, )* }
            }
        }
        impl ops::Rem<Scalar<$float>> for $vec<$float> {
            type Output = $vec<$float>;
            #[inline]
            fn rem(self, rhs: Scalar<$float>) -> Self::Output {
                $vec { $( $component: self.$component % rhs.0, )* }
            }
        }
    }
}

/// Bitwise impls for vectors of integers and vectors of bools
macro_rules! impl_vector_bitwise_with_bool {
    ($vec:ident, $int:ty, $( $component:tt )*) => {
        impl ops::BitAnd for $vec<$int> {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component & rhs.$component, )* }
            }
        }
        impl ops::BitOr for $vec<$int> {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component | rhs.$component, )* }
            }
        }
        impl ops::BitXor for $vec<$int> {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self::Output {
                $vec { $( $component: self.$component ^ rhs.$component, )* }
            }
        }
        impl ops::Not for $vec<$int> {
            type Output = Self;
            fn not(self) -> Self::Output {
                $vec { $( $component: !self.$component, )* }
            }
        }
    }
}

/// Bitwise impls for vectors of integers, not bools
macro_rules! impl_vector_bitwise_without_bool {
    ($vec:ident, $int:ty, $( $component:tt )*) => {
        impl_vector_bitwise_with_bool!($vec, $int, $($component)*);

        impl ops::Shl<$vec<u32>> for $vec<$int> {
            type Output = Self;
            fn shl(self, rhs: $vec<u32>) -> Self::Output {
                $vec { $( $component: self.$component.unbounded_shl(rhs.$component), )* }
            }
        }
        impl ops::Shr<$vec<u32>> for $vec<$int> {
            type Output = Self;
            fn shr(self, rhs: $vec<u32>) -> Self::Output {
                $vec { $( $component: self.$component.unbounded_shr(rhs.$component), )* }
            }
        }

    }
}

macro_rules! impl_element_casts {
    ($ty:ident) => {
        impl_element_casts!($ty, []);
    };
    ($ty:ident, [$($extra:tt)*]) => {
        // TODO: These do not have the right cast semantics, but what *are* the right cast
        // semantics? Naga IR docs are cryptic for `Expression::As`.
        pub fn cast_elem_as_u32(self) -> $ty<u32> {
            self.map(|component| component $($extra)* as u32)
        }
        pub fn cast_elem_as_i32(self) -> $ty<i32> {
            self.map(|component| component $($extra)* as i32)
        }
        pub fn cast_elem_as_f32(self) -> $ty<f32> {
            self.map(|component| component $($extra)* as f32)
        }
        pub fn cast_elem_as_f64(self) -> $ty<f64> {
            self.map(|component| component $($extra)* as f64)
        }
    };
}

/// Implement `Index` and `IndexMut`.
macro_rules! impl_index {
    ($component_count:literal, $vector_type:ident, $index_type:ty) => {
        impl<T> ops::Index<$index_type> for $vector_type<T> {
            type Output = Scalar<T>;

            #[inline]
            fn index(&self, index: $index_type) -> &Self::Output {
                // manual bounds check because we need to convert to usize and we’d like to have
                // only one panic branch rather than two
                if (0..$component_count).contains(&index) {
                    &self.as_array_ref()[index as usize]
                } else {
                    panic!("matrix indexing out of bounds")
                }
            }
        }

        impl<T> ops::IndexMut<$index_type> for $vector_type<T> {
            #[inline]
            fn index_mut(&mut self, index: $index_type) -> &mut Self::Output {
                if (0..$component_count).contains(&index) {
                    &mut self.as_array_mut()[index as usize]
                } else {
                    panic!("vector indexing out of bounds")
                }
            }
        }
    };
}

macro_rules! impl_vector_regular_fns {
    ( $ty:ident $component_count:literal : $( $component:tt )* ) => {
        impl<T> $ty<T> {
            pub fn splat(value: T) -> Self
            where
                T: Copy
            {
                Self { $( $component: value, )* }
            }
            pub fn splat_from_scalar(value: Scalar<T>) -> Self
            where
                T: Copy
            {
                Self { $( $component: value.0, )* }
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

            paste::paste! {
                $(
                    pub fn [< set_ $component >](&mut self, value: Scalar<T>) {
                        self.$component = value.0;
                    }
                )*
            }

            #[inline]
            fn as_array_ref(&self) -> &[Scalar<T>; $component_count] {
                // Reinterpret the reference to self as a reference to an array.
                // SAFETY: Vectors are `repr(C)` and have the same elements as the array.
                unsafe { &*(&raw const *self).cast::<[Scalar<T>; $component_count]>() }
            }
            #[inline]
            fn as_array_mut(&mut self) -> &mut [Scalar<T>; $component_count] {
                // Reinterpret the reference to self as a reference to an array.
                // SAFETY: Vectors are `repr(C)` and have the same elements as the array.
                unsafe { &mut *(&raw mut *self).cast::<[Scalar<T>; $component_count]>() }
            }

            /// As per WGSL [`dot()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#dot-builtin).
            //---
            // Design note: this is generic so that it can be used by generic matrix operations
            #[inline]
            pub fn dot(self, rhs: Self) -> Scalar<T>
            where
                // TODO: the ConstZero is purely an artifact of the macro repetition
                Scalar<T>: ops::Mul<Output = Scalar<T>> + num_traits::ConstZero,
            {
                $( Scalar(self.$component) * Scalar(rhs.$component) + )* Scalar::<T>::ZERO
            }

        }

        impl_vector_integer_arithmetic!($ty, i32, $($component)*);
        impl_vector_integer_arithmetic!($ty, u32, $($component)*);
        impl_vector_float_arithmetic!($ty, f32, $($component)*);
        impl_vector_float_arithmetic!($ty, f64, $($component)*);

        impl_vector_bitwise_with_bool!($ty, bool, $($component)*);
        impl_vector_bitwise_without_bool!($ty, i32, $($component)*);
        impl_vector_bitwise_without_bool!($ty, u32, $($component)*);

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
        impl $ty<bool> {
            impl_element_casts!($ty, [as u8]);
        }

        // Indexing, by usize, i32, or u32, yields a scalar
        impl_index!($component_count, $ty, usize);
        impl_index!($component_count, $ty, i32);
        impl_index!($component_count, $ty, u32);

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
        impl<T> From<$ty<T>> for [T; $component_count] {
            fn from(value: $ty<T>) -> Self {
                [$( value.$component ),*]
            }
        }

        // Irregular integer math: `$vec<u32>.abs()` exists even though it is the identity,
        // but Rust doesn't have it.
        impl $ty<i32> {
            delegate_unary_methods_elementwise!(const { abs } ($($component)*));
        }
        impl $ty<u32> {
            pub fn abs(self) -> Self { self }
        }

        // Boolean-specific functions
        impl $ty<bool> {
            /// As per WGSL [`any()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#any-builtin).
            pub fn any(self) -> Scalar<bool> {
                Scalar($( self.$component | )* false)
            }
            /// As per WGSL [`all()`](https://www.w3.org/TR/2025/CRD-WGSL-20250322/#all-builtin).
            pub fn all(self) -> Scalar<bool> {
                Scalar($( self.$component & )* true)
            }
        }
    }
}

impl_vector_regular_fns!(Scalar 1 : 0);
impl_vector_regular_fns!(Vec2 2 : x y);
impl_vector_regular_fns!(Vec3 3 : x y z);
impl_vector_regular_fns!(Vec4 4 : x y z w);

// -------------------------------------------------------------------------------------------------
// Impls that differ between `Vec*` and `Scalar`

/// Applied to every `Vec` type but not `Scalar`
macro_rules! impl_vector_not_scalar_fns {
    ( $ty:ident $component_count:literal : $( $component:tt )* ) => {
        impl<T> $ty<T> {
            // User-friendly constructor, not used by translated code.
            pub const fn new($( $component: T, )*) -> Self {
                Self { $( $component, )* }
            }

            pub const fn from_scalars($( $component: Scalar<T>, )*) -> Self
            where
                T: Copy
            {
                Self { $( $component: $component.0, )* }
            }
        }

        impl<T> From<[T; $component_count]> for $ty<T> {
            fn from(value: [T; $component_count]) -> Self {
                let [$( $component ),*] = value;
                Self::new($( $component ),*)
            }
        }

        impl_vector_scalar_integer_arithmetic!($ty, i32, $($component)*);
        impl_vector_scalar_integer_arithmetic!($ty, u32, $($component)*);
        impl_vector_scalar_float_arithmetic!($ty, f32, $($component)*);
        impl_vector_scalar_float_arithmetic!($ty, f64, $($component)*);

        // Commutative scalar-vector operators are derived from vector-scalar operators
        impl<T> ops::Add<$ty<T>> for Scalar<T>
        where
            $ty<T>: ops::Add<Scalar<T>>
        {
            type Output = <$ty<T> as ops::Add<Scalar<T>>>::Output;
            fn add(self, rhs: $ty<T>) -> Self::Output {
                rhs + self
            }
        }
        impl<T> ops::Mul<$ty<T>> for Scalar<T>
        where
            $ty<T>: ops::Mul<Scalar<T>>
        {
            type Output = <$ty<T> as ops::Mul<Scalar<T>>>::Output;
            fn mul(self, rhs: $ty<T>) -> Self::Output {
                rhs * self
            }
        }

    }
}

impl_vector_not_scalar_fns!(Vec2 2 : x y);
impl_vector_not_scalar_fns!(Vec3 3 : x y z);
impl_vector_not_scalar_fns!(Vec4 4 : x y z w);

impl<T> Scalar<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }
}
impl<T> From<[T; 1]> for Scalar<T> {
    fn from([value]: [T; 1]) -> Self {
        Self(value)
    }
}

// These impls must be per-primitive-type to pass trait coherence checking.
macro_rules! impl_from_scalar_to_inner {
    ($t:ty) => {
        impl From<Scalar<$t>> for $t {
            fn from(value: Scalar<$t>) -> Self {
                value.0
            }
        }
    };
}
impl_from_scalar_to_inner!(bool);
impl_from_scalar_to_inner!(i32);
impl_from_scalar_to_inner!(u32);
impl_from_scalar_to_inner!(f32);
impl_from_scalar_to_inner!(f64);

// -------------------------------------------------------------------------------------------------
// Irregular functions and impls

impl<T> Scalar<T> {
    /// Currently equivalent to `self.0` but can be used as a more strongly typed operation.
    pub fn into_inner(self) -> T {
        self.0
    }

    pub fn into_array_index(self) -> usize
    where
        T: Copy + TryInto<usize> + fmt::Debug,
    {
        match self.into_inner().try_into() {
            Ok(index) => index,
            Err(_) => panic!("impossible array index {self:?}"),
        }
    }
}

impl Scalar<bool> {
    /// Returns the boolean value of this scalar.
    ///
    /// This method only exists for `Scalar<bool>`.
    // (In the future when we support SIMD, this will be unavailable in the SIMD mode.)
    pub fn into_branch_condition(self) -> bool {
        self.0
    }
}

impl<T> From<T> for Scalar<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T: Default> Default for Scalar<T> {
    fn default() -> Self {
        Self(Default::default())
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
    impl_flattening_ctor!(fn new_12(x: Scalar<T>, yz: Vec2<T>) => (x.0, yz.x, yz.y));
    impl_flattening_ctor!(fn new_21(xy: Vec2<T>, z: Scalar<T>) => (xy.x, xy.y, z.0));
}
impl<T: Copy> Vec4<T> {
    impl_flattening_ctor!(fn new_112(x: Scalar<T>, y: Scalar<T>, zw: Vec2<T>) => (x.0, y.0, zw.x, zw.y));
    impl_flattening_ctor!(fn new_121(x: Scalar<T>, yz: Vec2<T>, w: Scalar<T>) => (x.0, yz.x, yz.y, w.0));
    impl_flattening_ctor!(fn new_211(xy: Vec2<T>, z: Scalar<T>, w: Scalar<T>) => (xy.x, xy.y, z.0, w.0));
    impl_flattening_ctor!(fn new_22(xy: Vec2<T>, zw: Vec2<T>) => (xy.x, xy.y, zw.x, zw.y));
    impl_flattening_ctor!(fn new_13(x: Scalar<T>, yzw: Vec3<T>) => (x.0, yzw.x, yzw.y, yzw.z));
    impl_flattening_ctor!(fn new_31(xyz: Vec3<T>, w: Scalar<T>) => (xyz.x, xyz.y, xyz.z, w.0));
}

// -------------------------------------------------------------------------------------------------
// Swizzles and element accessors

macro_rules! scalar_accessors {
    ($get:ident $get_mut:ident $component:ident ) => {
        /// Returns the
        #[doc = stringify!($component)]
        /// component of `self` as a [`Scalar`].
        pub fn $get(self) -> Scalar<T> {
            Scalar(self.$component)
        }

        /// Returns a reference to the
        #[doc = stringify!($component)]
        /// component of `self` as a [`Scalar`].
        pub fn $get_mut(&mut self) -> &mut Scalar<T> {
            // SAFETY: `Scalar` is `repr(transparent)`
            unsafe { &mut *(&raw mut self.$component).cast::<Scalar<T>>() }
        }
    };
}

macro_rules! swizzle_fn {
    ($name:ident $output:ident ($($cin:ident)*) ) => {
        /// Takes the
        #[doc = stringify!($($cin),*)]
        /// elements of `self` and returns them in that order.
        pub fn $name(self) -> $output<T> {
            $output::new($(self.$cin,)*)
        }
    }
}

impl<T: Copy> Vec2<T> {
    scalar_accessors!(x x_mut x);
    scalar_accessors!(y y_mut y);
    swizzle_fn!(xy Vec2(x y));
    swizzle_fn!(yx Vec2(y x));
}
impl<T: Copy> Vec3<T> {
    scalar_accessors!(x x_mut x);
    scalar_accessors!(y y_mut y);
    scalar_accessors!(z z_mut z);
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
    scalar_accessors!(x x_mut x);
    scalar_accessors!(y y_mut y);
    scalar_accessors!(z z_mut z);
    scalar_accessors!(w w_mut w);
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
