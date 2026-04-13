use core::ops;

use crate::{Scalar, Vec2, Vec3, Vec4};

// -------------------------------------------------------------------------------------------------

/// Generate a row vector from a matrix.
/// Has to be a separate macro due to the inexpressiveness of macro repetitions.
macro_rules! generate_row_expr {
    ($self:ident, $vec_type:ident, $row_field:ident, [$($column_field:ident),*]) => {
        $vec_type::new($( $self.$column_field.$row_field ),*)
    }
}

/// Generate the body of a transpose() method.
/// Has to be a separate macro due to the inexpressiveness of macro repetitions.
macro_rules! transpose_body {
    ($self:ident, $mat_type:ident, $vec_type:ident, [$($row_field:ident),*], $column_fields:tt) => {
        $mat_type {
            $(
                $row_field: generate_row_expr!($self, $vec_type, $row_field, $column_fields),
            )*
        }
    }
}

/// Implement `Index` and `IndexMut`.
macro_rules! impl_index {
    ($columns:literal, $mat_type:ident, $column_type:ident, $index_type:ty) => {
        impl<T> ops::Index<$index_type> for $mat_type<T> {
            type Output = $column_type<T>;

            #[inline]
            fn index(&self, index: $index_type) -> &Self::Output {
                // manual bounds check because we need to convert to usize and we’d like to have
                // only one panic branch rather than two
                if (0..$columns).contains(&index) {
                    &self.as_array_of_columns_ref()[index as usize]
                } else {
                    panic!("matrix indexing out of bounds")
                }
            }
        }

        impl<T> ops::IndexMut<$index_type> for $mat_type<T> {
            #[inline]
            fn index_mut(&mut self, index: $index_type) -> &mut Self::Output {
                if (0..$columns).contains(&index) {
                    &mut self.as_array_of_columns_mut()[index as usize]
                } else {
                    panic!("matrix indexing out of bounds")
                }
            }
        }
    };
}

macro_rules! matrix_struct {
    ($columns:literal, $rows:literal, $column_type:ident, [$($column_field:ident),*], $row_type:ident, [$($row_field:ident),*]) => {
        paste::paste! {
            #[doc = concat!("Matrix with ", $columns, " columns and ", $rows, " rows.")]
            ///
            /// The matrix is stored column-major; that is, each field is a whole column of the matrix.
            #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
            #[repr(C)]
            pub struct [< Mat $columns x $rows >] <T> {
                $( pub $column_field: $column_type<T>, )*
            }

            impl<T> [< Mat $columns x $rows >] <T> {
                pub fn new($([< $column_field _column >]: $column_type<T>,)*) -> Self {
                    Self { $($column_field: [< $column_field _column >],)* }
                }

                pub fn transpose(self) -> [< Mat $rows x $columns >] <T> {
                    transpose_body!(
                        self,
                        [< Mat $rows x $columns >],
                        $row_type,
                        [$($row_field),*],
                        [$($column_field),*]
                    )
                }

                #[inline]
                fn as_array_of_columns_ref(&self) -> &[$column_type<T>; $columns] {
                    // Reinterpret the reference to self as a reference to an array.
                    // SAFETY: Matrices are `repr(C)` and have the same elements as the array.
                    unsafe { &*(&raw const *self).cast::<[$column_type<T>; $columns]>() }
                }
                #[inline]
                fn as_array_of_columns_mut(&mut self) -> &mut [$column_type<T>; $columns] {
                    // Reinterpret the reference to self as a reference to an array.
                    // SAFETY: Matrices are `repr(C)` and have the same elements as the array.
                    unsafe { &mut *(&raw mut *self).cast::<[$column_type<T>; $columns]>() }
                }
            }

            // Indexing, by usize, i32, or u32, yields a column vector
            impl_index!($columns, [< Mat $columns x $rows >], $column_type, usize);
            impl_index!($columns, [< Mat $columns x $rows >], $column_type, i32);
            impl_index!($columns, [< Mat $columns x $rows >], $column_type, u32);

            impl<T> ops::Add for [< Mat $columns x $rows >]<T>
            where
                $column_type<T>: ops::Add<Output = $column_type<T>>,
            {
                type Output = Self;

                /// Performs component-wise addition.
                #[inline]
                fn add(self, rhs: Self) -> Self::Output {
                    Self::new(
                        $( self.$column_field + rhs.$column_field ),*
                    )
                }
            }

            impl<T> ops::Sub for [< Mat $columns x $rows >]<T>
            where
                $column_type<T>: ops::Sub<Output = $column_type<T>>,
            {
                type Output = Self;

                /// Performs component-wise subtraction.
                #[inline]
                fn sub(self, rhs: Self) -> Self::Output {
                    Self::new(
                        $( self.$column_field - rhs.$column_field ),*
                    )
                }
            }

            impl<T> ops::Mul<Scalar<T>> for [< Mat $columns x $rows >]<T>
            where
                $column_type<T>: ops::Mul<Scalar<T>, Output = $column_type<T>>,
                T: Copy,
            {
                type Output = Self;

                /// Performs component-wise multiplication by a scalar.
                #[inline]
                fn mul(self, rhs: Scalar<T>) -> Self::Output {
                    Self::new(
                        $( self.$column_field * rhs ),*
                    )
                }
            }

            impl<T> ops::Mul<[< Mat $columns x $rows >]<T>> for Scalar<T>
            where
                Scalar<T>: ops::Mul<$column_type<T>, Output = $column_type<T>>,
                T: Copy,
            {
                type Output = [< Mat $columns x $rows >]<T>;

                /// Performs component-wise multiplication by a scalar.
                #[inline]
                fn mul(self, rhs: [< Mat $columns x $rows >]<T>) -> Self::Output {
                    Self::Output::new(
                        $( self * rhs.$column_field ),*
                    )
                }
            }

            impl<T> ops::Mul<$row_type<T>> for [< Mat $columns x $rows >]<T>
            where
                // bounds copied from dot()
                Scalar<T>: ops::Mul<Output = Scalar<T>> + num_traits::ConstZero,
                T: Copy,
            {
                type Output = $column_type<T>;

                /// Multiplication with matrix on the left and vector on the right.
                #[inline]
                fn mul(self, rhs: $row_type<T>) -> Self::Output {
                    let t = self.transpose();
                    $column_type::from_scalars(
                        // dot product of LHS rows with RHS column
                        $( t.$row_field.dot(rhs) ),*
                    )
                }
            }

            impl<T> ops::Mul<[< Mat $columns x $rows >]<T>> for $column_type<T>
            where
                Scalar<T>: ops::Mul<Output = Scalar<T>> + num_traits::ConstZero,
                T: Copy,
            {
                type Output = $row_type<T>;

                /// Multiplication with vector on the left and matrix on the right.
                #[inline]
                fn mul(self, rhs: [< Mat $columns x $rows >]<T>) -> Self::Output {
                    $row_type::from_scalars(
                        // dot product of LHS row with RHS columns
                        $( self.dot(rhs.$column_field) ),*
                    )
                }
            }
        }
    }
}

matrix_struct!(2, 2, Vec2, [x, y], Vec2, [x, y]);
matrix_struct!(2, 3, Vec3, [x, y], Vec2, [x, y, z]);
matrix_struct!(2, 4, Vec4, [x, y], Vec2, [x, y, z, w]);
matrix_struct!(3, 2, Vec2, [x, y, z], Vec3, [x, y]);
matrix_struct!(3, 3, Vec3, [x, y, z], Vec3, [x, y, z]);
matrix_struct!(3, 4, Vec4, [x, y, z], Vec3, [x, y, z, w]);
matrix_struct!(4, 2, Vec2, [x, y, z, w], Vec4, [x, y]);
matrix_struct!(4, 3, Vec3, [x, y, z, w], Vec4, [x, y, z]);
matrix_struct!(4, 4, Vec4, [x, y, z, w], Vec4, [x, y, z, w]);

// -------------------------------------------------------------------------------------------------

macro_rules! matrix_multiply {
    (
        $rows:literal,
        $columns:literal,
        $common:literal,
        [$($column_field:ident),*]
    ) => {
        paste::paste! {
            impl ops::Mul<[< Mat $columns x $common >]<f32>> for [< Mat $common x $rows >]<f32> {
                type Output = [< Mat $columns x $rows >]<f32>;

                /// Performs matrix multiplication.
                #[inline]
                fn mul(self, rhs: [< Mat $columns x $common >]<f32>) -> Self::Output {
                    [< Mat $columns x $rows >]::new(
                        $( self * rhs.$column_field ),*
                    )
                }
            }

        }
    }
}

matrix_multiply!(2, 2, 2, [x, y]);
matrix_multiply!(2, 2, 3, [x, y]);
matrix_multiply!(2, 2, 4, [x, y]);
matrix_multiply!(2, 3, 2, [x, y, z]);
matrix_multiply!(2, 3, 3, [x, y, z]);
matrix_multiply!(2, 3, 4, [x, y, z]);
matrix_multiply!(2, 4, 2, [x, y, z, w]);
matrix_multiply!(2, 4, 3, [x, y, z, w]);
matrix_multiply!(2, 4, 4, [x, y, z, w]);
matrix_multiply!(3, 2, 2, [x, y]);
matrix_multiply!(3, 2, 3, [x, y]);
matrix_multiply!(3, 2, 4, [x, y]);
matrix_multiply!(3, 3, 2, [x, y, z]);
matrix_multiply!(3, 3, 3, [x, y, z]);
matrix_multiply!(3, 3, 4, [x, y, z]);
matrix_multiply!(3, 4, 2, [x, y, z, w]);
matrix_multiply!(3, 4, 3, [x, y, z, w]);
matrix_multiply!(3, 4, 4, [x, y, z, w]);
matrix_multiply!(4, 2, 2, [x, y]);
matrix_multiply!(4, 2, 3, [x, y]);
matrix_multiply!(4, 2, 4, [x, y]);
matrix_multiply!(4, 3, 2, [x, y, z]);
matrix_multiply!(4, 3, 3, [x, y, z]);
matrix_multiply!(4, 3, 4, [x, y, z]);
matrix_multiply!(4, 4, 2, [x, y, z, w]);
matrix_multiply!(4, 4, 3, [x, y, z, w]);
matrix_multiply!(4, 4, 4, [x, y, z, w]);
