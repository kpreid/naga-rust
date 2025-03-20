/*!
Conversion of Naga/shader vocabulary to Rust.
*/

use core::fmt;

use naga::{MathFunction, Scalar};

/// Local path to the module which is the “standard library” for shader functionality
/// that doesn't map directly to Rust `core`.
///
/// To keep the output concise, this is not the crate name,
/// but is guaranteed to be provided via a `use` in the same scope.
pub(crate) const SHADER_LIB: &str = "rt";

/// Types that can return the Rust source representation of their
/// values as a `'static` string.
///
/// This trait is specifically for types whose Rust forms are simple
/// enough that they can always be returned as a static string.
pub(crate) trait ToRust: Sized {
    /// Return Rust source code representation of `self`.
    fn to_rust(self) -> &'static str;
}

/// Types that may be able to return the Rust source representation
/// for their values as a `'static` string.
///
/// This trait is specifically for types whose values are either
/// simple enough that their Rust form can be represented a static
/// string, or aren't representable in Rust at all.
///
/// - If all values in the type have `&'static str` representations in
///   Rust, consider implementing [`ToRust`] instead.
///
/// - If a type's Rust form requires dynamic formatting, so that
///   returning a `&'static str` isn't feasible, consider implementing
///   [`core::fmt::Display`] on some wrapper type instead.
pub trait TryToRust: Sized {
    /// Return the Rust form of `self` as a `'static` string.
    ///
    /// If `self` doesn't have a representation in Rust, then return `None`.
    fn try_to_rust(self) -> Option<&'static str>;

    /// What kind of Rust thing `Self` represents.
    const DESCRIPTION: &'static str;
}

pub(crate) fn unwrap_to_rust<T: TryToRust + Copy + fmt::Debug>(value: T) -> &'static str {
    value.try_to_rust().unwrap_or_else(|| {
        unreachable!(
            "validation should have forbidden {}: {value:?}",
            T::DESCRIPTION
        );
    })
}

// Contains all keywords, strict or weak, in 2024 and any previous edition, sorted.
// https://doc.rust-lang.org/reference/keywords.html
pub const KEYWORDS_2024: &[&str] = &[
    "abstract",
    "as",
    "async",
    "await",
    "become",
    "box",
    "break",
    "const",
    "continue",
    "crate",
    "do",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "final",
    "fn",
    "for",
    "gen",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "macro_rules",
    "macro",
    "match",
    "mod",
    "move",
    "mut",
    "override",
    "priv",
    "pub",
    "raw",
    "ref",
    "return",
    "safe",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "try",
    "type",
    "typeof",
    "union",
    "unsafe",
    "unsized",
    "use",
    "virtual",
    "where",
    "while",
    "yield",
];

impl ToRust for MathFunction {
    fn to_rust(self) -> &'static str {
        use MathFunction as Mf;
        match self {
            Mf::Abs => "rt::abs",
            Mf::Min => "rt::min",
            Mf::Max => "rt::max",
            Mf::Clamp => "rt::clamp",
            Mf::Saturate => "rt::saturate",
            Mf::Cos => "rt::cos",
            Mf::Cosh => "rt::cosh",
            Mf::Sin => "rt::sin",
            Mf::Sinh => "rt::sinh",
            Mf::Tan => "rt::tan",
            Mf::Tanh => "rt::tanh",
            Mf::Acos => "rt::acos",
            Mf::Asin => "rt::asin",
            Mf::Atan => "rt::atan",
            Mf::Atan2 => "rt::atan2",
            Mf::Asinh => "rt::asinh",
            Mf::Acosh => "rt::acosh",
            Mf::Atanh => "rt::atanh",
            Mf::Radians => "rt::radians",
            Mf::Degrees => "rt::degrees",
            Mf::Ceil => "rt::ceil",
            Mf::Floor => "rt::floor",
            Mf::Round => "rt::round",
            Mf::Fract => "rt::fract",
            Mf::Trunc => "rt::trunc",
            Mf::Modf => "rt::modf",
            Mf::Frexp => "rt::frexp",
            Mf::Ldexp => "rt::ldexp",
            Mf::Exp => "rt::exp",
            Mf::Exp2 => "rt::exp2",
            Mf::Log => "rt::log",
            Mf::Log2 => "rt::log2",
            Mf::Pow => "rt::pow",
            Mf::Dot => "rt::dot",
            Mf::Cross => "rt::cross",
            Mf::Distance => "rt::distance",
            Mf::Length => "rt::length",
            Mf::Normalize => "rt::normalize",
            Mf::FaceForward => "rt::faceForward",
            Mf::Reflect => "rt::reflect",
            Mf::Refract => "rt::refract",
            Mf::Sign => "rt::sign",
            Mf::Fma => "rt::fma",
            Mf::Mix => "rt::mix",
            Mf::Step => "rt::step",
            Mf::SmoothStep => "rt::smoothstep",
            Mf::Sqrt => "rt::sqrt",
            Mf::InverseSqrt => "rt::inverseSqrt",
            Mf::Transpose => "rt::transpose",
            Mf::Determinant => "rt::determinant",
            Mf::QuantizeToF16 => "rt::quantizeToF16",
            Mf::CountTrailingZeros => "rt::countTrailingZeros",
            Mf::CountLeadingZeros => "rt::countLeadingZeros",
            Mf::CountOneBits => "rt::countOneBits",
            Mf::ReverseBits => "rt::reverseBits",
            Mf::ExtractBits => "rt::extractBits",
            Mf::InsertBits => "rt::insertBits",
            Mf::FirstTrailingBit => "rt::firstTrailingBit",
            Mf::FirstLeadingBit => "rt::firstLeadingBit",
            Mf::Pack4x8snorm => "rt::pack4x8snorm",
            Mf::Pack4x8unorm => "rt::pack4x8unorm",
            Mf::Pack2x16snorm => "rt::pack2x16snorm",
            Mf::Pack2x16unorm => "rt::pack2x16unorm",
            Mf::Pack2x16float => "rt::pack2x16float",
            Mf::Pack4xI8 => "rt::pack4xI8",
            Mf::Pack4xU8 => "rt::pack4xU8",
            Mf::Unpack4x8snorm => "rt::unpack4x8snorm",
            Mf::Unpack4x8unorm => "rt::unpack4x8unorm",
            Mf::Unpack2x16snorm => "rt::unpack2x16snorm",
            Mf::Unpack2x16unorm => "rt::unpack2x16unorm",
            Mf::Unpack2x16float => "rt::unpack2x16float",
            Mf::Unpack4xI8 => "rt::unpack4xI8",
            Mf::Unpack4xU8 => "rt::unpack4xU8",
            Mf::Outer => "rt::outer",
            Mf::Inverse => "rt::inverse",
        }
    }
}

impl TryToRust for Scalar {
    const DESCRIPTION: &'static str = "scalar type";

    fn try_to_rust(self) -> Option<&'static str> {
        Some(match self {
            Scalar::F64 => "f64",
            Scalar::F32 => "f32",
            Scalar::I32 => "i32",
            Scalar::U32 => "u32",
            Scalar::I64 => "i64",
            Scalar::U64 => "u64",
            Scalar::BOOL => "bool",
            _ => return None,
        })
    }
}

/// Prefix used in type names.
pub fn upper_glam_prefix(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::F32 => "",
        Scalar::F64 => "D",
        Scalar::I32 => "I",
        Scalar::U32 => "U",
        Scalar::I64 => "I64",
        Scalar::U64 => "U64",
        Scalar::BOOL => "B",
        _ => unreachable!(),
    }
}
/// Prefix used in function names.
pub fn lower_glam_prefix(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::F32 => "",
        Scalar::F64 => "d",
        Scalar::I32 => "i",
        Scalar::U32 => "u",
        Scalar::I64 => "i64",
        Scalar::U64 => "u64",
        Scalar::BOOL => "b",
        _ => unreachable!(),
    }
}

impl ToRust for naga::Interpolation {
    fn to_rust(self) -> &'static str {
        match self {
            naga::Interpolation::Perspective => "perspective",
            naga::Interpolation::Linear => "linear",
            naga::Interpolation::Flat => "flat",
        }
    }
}

impl ToRust for naga::Sampling {
    fn to_rust(self) -> &'static str {
        match self {
            naga::Sampling::Center => "center",
            naga::Sampling::Centroid => "centroid",
            naga::Sampling::Sample => "sample",
            naga::Sampling::First => "first",
            naga::Sampling::Either => "either",
        }
    }
}

/// Grouping binary operators into categories which determine what kind of Rust code
/// must be generated for them.
///
/// For example, the `<` operator cannot be overloaded as a vector operation,
/// because it is defined to always return `bool`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BinOpClassified {
    /// Can be overloaded to take a vector and return a vector.
    #[allow(dead_code, reason = "TODO: review whether the field should be read")]
    Vectorizable(BinOpVec),
    /// Always returns `bool`.
    ScalarBool(BinOpBool),
}

/// Part of [`BinOpClassified`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum BinOpVec {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    And,
    ExclusiveOr,
    InclusiveOr,
    ShiftLeft,
    ShiftRight,
}

/// Part of [`BinOpClassified`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum BinOpBool {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
}

impl From<naga::BinaryOperator> for BinOpClassified {
    fn from(value: naga::BinaryOperator) -> Self {
        use BinOpClassified as C;
        use naga::BinaryOperator as Bo;
        match value {
            Bo::Add => C::Vectorizable(BinOpVec::Add),
            Bo::Subtract => C::Vectorizable(BinOpVec::Subtract),
            Bo::Multiply => C::Vectorizable(BinOpVec::Multiply),
            Bo::Divide => C::Vectorizable(BinOpVec::Divide),
            Bo::Modulo => C::Vectorizable(BinOpVec::Modulo),
            Bo::And => C::Vectorizable(BinOpVec::And),
            Bo::ExclusiveOr => C::Vectorizable(BinOpVec::ExclusiveOr),
            Bo::InclusiveOr => C::Vectorizable(BinOpVec::InclusiveOr),
            Bo::Equal => C::ScalarBool(BinOpBool::Equal),
            Bo::NotEqual => C::ScalarBool(BinOpBool::NotEqual),
            Bo::Less => C::ScalarBool(BinOpBool::Less),
            Bo::LessEqual => C::ScalarBool(BinOpBool::LessEqual),
            Bo::Greater => C::ScalarBool(BinOpBool::Greater),
            Bo::GreaterEqual => C::ScalarBool(BinOpBool::GreaterEqual),
            Bo::LogicalAnd => C::ScalarBool(BinOpBool::LogicalAnd),
            Bo::LogicalOr => C::ScalarBool(BinOpBool::LogicalOr),
            Bo::ShiftLeft => C::Vectorizable(BinOpVec::ShiftLeft),
            Bo::ShiftRight => C::Vectorizable(BinOpVec::ShiftRight),
        }
    }
}

impl BinOpBool {
    pub fn to_vector_fn(self) -> &'static str {
        use BinOpBool as Bo;
        match self {
            Bo::Equal => "v_eq",
            Bo::NotEqual => "v_ne",
            Bo::Less => "v_lt",
            Bo::LessEqual => "v_le",
            Bo::Greater => "v_gt",
            Bo::GreaterEqual => "v_ge",
            Bo::LogicalAnd => "v_land",
            Bo::LogicalOr => "v_lor",
        }
    }
}

pub(crate) const fn vector_size_str(size: naga::VectorSize) -> &'static str {
    match size {
        naga::VectorSize::Bi => "2",
        naga::VectorSize::Tri => "3",
        naga::VectorSize::Quad => "4",
    }
}
