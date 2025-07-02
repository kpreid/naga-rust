/*!
Conversion of Naga/shader vocabulary to Rust.
*/

use alloc::boxed::Box;
use alloc::format;
use core::fmt;

use naga::Scalar;

/// Types that may be able to return the Rust source representation
/// for their values as a `'static` string.
///
/// This trait is specifically for types whose values are either
/// simple enough that their Rust form can be represented a static
/// string, or aren't representable in Rust at all.
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
const KEYWORDS_2024_SLICE: &[&str] = &[
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

pub(crate) fn keywords_2024() -> &'static hashbrown::HashSet<&'static str> {
    use once_cell::race::OnceBox;
    static KEYWORDS: OnceBox<hashbrown::HashSet<&'static str>> = OnceBox::new();
    KEYWORDS.get_or_init(|| Box::new(KEYWORDS_2024_SLICE.iter().copied().collect()))
}

/// Converts a [`MathFunction`] to a Rust method name.
///
/// These methods are implemented on `naga_rust_rt` vector types, and also overlap
/// with Rust `std` functions on scalars.
/// TODO: But not all of them do, so this won’t fully work correctly until  `naga_rust_rt::Scalar`
/// is in use.
pub(crate) fn math_function_to_method(f: naga::MathFunction) -> &'static str {
    use naga::MathFunction as Mf;
    match f {
        Mf::Abs => "abs",
        Mf::Min => "min",
        Mf::Max => "max",
        Mf::Clamp => "clamp",
        Mf::Saturate => "saturate",
        Mf::Cos => "cos",
        Mf::Cosh => "cosh",
        Mf::Sin => "sin",
        Mf::Sinh => "sinh",
        Mf::Tan => "tan",
        Mf::Tanh => "tanh",
        Mf::Acos => "acos",
        Mf::Asin => "asin",
        Mf::Atan => "atan",
        Mf::Atan2 => "atan2",
        Mf::Asinh => "asinh",
        Mf::Acosh => "acosh",
        Mf::Atanh => "atanh",
        Mf::Radians => "to_radians",
        Mf::Degrees => "to_degrees",
        Mf::Ceil => "ceil",
        Mf::Floor => "floor",
        Mf::Round => "round",
        Mf::Fract => "fract",
        Mf::Trunc => "trunc",
        Mf::Modf => "modf",
        Mf::Frexp => "frexp",
        Mf::Ldexp => "ldexp",
        Mf::Exp => "exp",
        Mf::Exp2 => "exp2",
        Mf::Log => "log",
        Mf::Log2 => "log2",
        Mf::Pow => "powf",
        Mf::Dot => "dot",
        Mf::Cross => "cross",
        Mf::Distance => "distance",
        Mf::Length => "length",
        Mf::Normalize => "normalize",
        Mf::FaceForward => "face_forward",
        Mf::Reflect => "reflect",
        Mf::Refract => "refract",
        Mf::Sign => "sign",
        Mf::Fma => "mul_add",
        Mf::Mix => "mix",
        Mf::Step => "step",
        Mf::SmoothStep => "smoothstep",
        Mf::Sqrt => "sqrt",
        Mf::InverseSqrt => "inverse_sqrt",
        Mf::Transpose => "transpose",
        Mf::Determinant => "determinant",
        // TODO: rename these to Rust style
        Mf::QuantizeToF16 => "quantizeToF16",
        Mf::CountTrailingZeros => "countTrailingZeros",
        Mf::CountLeadingZeros => "countLeadingZeros",
        Mf::CountOneBits => "countOneBits",
        Mf::ReverseBits => "reverseBits",
        Mf::ExtractBits => "extractBits",
        Mf::InsertBits => "insertBits",
        Mf::FirstTrailingBit => "firstTrailingBit",
        Mf::FirstLeadingBit => "firstLeadingBit",
        Mf::Pack4x8snorm => "pack4x8snorm",
        Mf::Pack4x8unorm => "pack4x8unorm",
        Mf::Pack2x16snorm => "pack2x16snorm",
        Mf::Pack2x16unorm => "pack2x16unorm",
        Mf::Pack2x16float => "pack2x16float",
        Mf::Pack4xI8 => "pack4xI8",
        Mf::Pack4xU8 => "pack4xU8",
        Mf::Unpack4x8snorm => "unpack4x8snorm",
        Mf::Unpack4x8unorm => "unpack4x8unorm",
        Mf::Unpack2x16snorm => "unpack2x16snorm",
        Mf::Unpack2x16unorm => "unpack2x16unorm",
        Mf::Unpack2x16float => "unpack2x16float",
        Mf::Unpack4xI8 => "unpack4xI8",
        Mf::Unpack4xU8 => "unpack4xU8",
        Mf::Outer => "outer",
        Mf::Inverse => "inverse",
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

/// Maps a scalar type to the corresponding `core::sync::atomic` type.
pub fn atomic_type_name(scalar: Scalar) -> Result<&'static str, crate::Error> {
    Ok(match scalar {
        Scalar::I32 => "AtomicI32",
        Scalar::U32 => "AtomicU32",
        Scalar::I64 => "AtomicI64",
        Scalar::U64 => "AtomicU64",
        Scalar::BOOL => "AtomicBool",
        _ => return Err(crate::Error::Unimplemented(format!("atomic {scalar:?}"))),
    })
}

// TODO: This code won't be used until we support textures
//
// impl ToRust for naga::Interpolation {
//     fn to_rust(self) -> &'static str {
//         match self {
//             naga::Interpolation::Perspective => "perspective",
//             naga::Interpolation::Linear => "linear",
//             naga::Interpolation::Flat => "flat",
//         }
//     }
// }
//
// impl ToRust for naga::Sampling {
//     fn to_rust(self) -> &'static str {
//         match self {
//             naga::Sampling::Center => "center",
//             naga::Sampling::Centroid => "centroid",
//             naga::Sampling::Sample => "sample",
//             naga::Sampling::First => "first",
//             naga::Sampling::Either => "either",
//         }
//     }
// }

/// Grouping binary operators into categories which determine what kind of Rust code
/// must be generated for them.
///
/// For example, the `<` operator cannot be overloaded as a vector operation,
/// because it is defined to always return `bool`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BinOpClassified {
    /// The Rust operator can be overloaded to take a vector and return a vector.
    #[allow(dead_code, reason = "TODO: review whether the field should be read")]
    Vectorizable(BinOpVec),
    /// The Rust operator always returns `bool`, but we want to return vectors instead.
    ScalarBool(BinOpBool),
    /// The operator is a non-vector operator and affects control flow.
    ShortCircuit(BinOpSc),
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
/// The operators that in Rust cannot be overloaded to return vectors.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BinOpBool {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

/// Part of [`BinOpClassified`].
/// The operators that apply to scalars and affect control flow.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BinOpSc {
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
            Bo::LogicalAnd => C::ShortCircuit(BinOpSc::LogicalAnd),
            Bo::LogicalOr => C::ShortCircuit(BinOpSc::LogicalOr),
            Bo::ShiftLeft => C::Vectorizable(BinOpVec::ShiftLeft),
            Bo::ShiftRight => C::Vectorizable(BinOpVec::ShiftRight),
        }
    }
}

impl BinOpBool {
    pub fn to_vector_method(self) -> &'static str {
        use BinOpBool as Bo;
        match self {
            Bo::Equal => "elementwise_eq",
            Bo::NotEqual => "elementwise_ne",
            Bo::Less => "elementwise_lt",
            Bo::LessEqual => "elementwise_le",
            Bo::Greater => "elementwise_gt",
            Bo::GreaterEqual => "elementwise_ge",
        }
    }
}

impl BinOpSc {
    pub fn to_binary_operator(self) -> &'static str {
        match self {
            BinOpSc::LogicalAnd => "&&",
            BinOpSc::LogicalOr => "||",
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
