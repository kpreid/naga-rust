/*!
Conversion of Naga/shader vocabulary to Rust.
*/

use alloc::boxed::Box;

use naga::proc::KeywordSet;

// Contains all keywords, strict or weak, in 2024 and any previous edition, sorted.
// Also contains names we want to reserve for our own purposes.
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
    "resources", // not a Rust keyword; reserved for ourselves.
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

pub(crate) fn keywords_2024() -> &'static KeywordSet {
    use once_cell::race::OnceBox;
    static KEYWORDS: OnceBox<KeywordSet> = OnceBox::new();
    KEYWORDS.get_or_init(|| Box::new(KEYWORDS_2024_SLICE.iter().copied().collect()))
}

/// Converts a [`MathFunction`] to a Rust method name.
///
/// These methods are implemented on `naga_rust_rt` vector types, and also overlap
/// with Rust `std` functions on scalars.
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
        Mf::Dot4I8Packed => "dot4I8Packed",
        Mf::Dot4U8Packed => "dot4U8Packed",
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
        Mf::Pack4xI8Clamp => "pack4xI8Clamp",
        Mf::Pack4xU8Clamp => "pack4xU8Clamp",
    }
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
