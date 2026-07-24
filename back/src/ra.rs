//! **R**ust **A**ST for code generation.
//!
//! This is not a general purpose AST. Rather, it is designed to support exactly what
//! `naga-rust` needs to emit.
//! It is never processed in any way but to write it out to text.
//! It is nearly unnecessary, but allows the translation logic to be much simpler and more flexible
//! by not interleaving translation logic with writing the text and by allowing subexpressions to
//! be built before their context is.
//!
//! Some of the types in this module refer to Naga enums.
//! This is done only when there is exactly one, infallible, conversion possible.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use naga::VectorSize;
use naga::back::{self, INDENT};

use crate::config::WriterFlags;

// -------------------------------------------------------------------------------------------------

/// A set of generic parameters (of the only sorts we use).
#[derive(Clone, Copy)]
pub(crate) enum Generics {
    /// No generics. Prints as the empty string.
    None,
    /// `<'g>`
    LtG,
}

/// A Rust type.
#[derive(Clone)]
pub(crate) enum Type {
    /// `()`
    Unit,
    /// `Self`
    Self_,
    /// Instantiation of a generic type with 1 parameter that is defined in the runtime crate.
    RtGen(RtGen, Scalar),
    /// Rust atomic type, e.g. [`core::sync::atomic::AtomicU32`],
    Atomic(Scalar),
    /// Rust scalar type, e.g. [`u32`].
    BareScalar(Scalar),
    /// Texture/image handle; one of the structs from `naga_rust_rt::texture`.
    /// TODO: Fold this into the `RtGen` variant because it has the same shape?
    Texture {
        dim: naga::ImageDimension,
        multisampled: bool,
        arrayed: bool,
        storage_type: Box<Type>,
    },
    /// `dyn naga_rust_rt::texture::Read<...>`
    DynTextureRead {
        dim: naga::ImageDimension,
        scalar: Scalar,
    },
    /// Texture sampler.
    Sampler,
    /// Texture sampler.
    SamplerComparison,
    /// Rust array type.
    Array(Box<Type>, u32),
    /// Rust slice type.
    Slice(Box<Type>),
    /// Rust pointer type (reference or raw).
    Ptr(PtrKind, Box<Type>),
    /// Nominal type that is declared in the shader code (usually a `struct`).
    User(String, Generics),
    /// `impl Into<T>`.
    ImplInto(Box<Type>),
}

/// A Rust trait or derive macro name.
#[derive(Clone)]
pub(crate) enum Trait {
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    /// User-provided path.
    #[expect(dead_code, reason = "TODO: add user-specified derives")]
    User(String),
}

/// A Rust pointer type, except for the pointee type.
#[derive(Clone, Copy)]
pub(crate) enum PtrKind {
    /// `&`, possibly with a lifetime
    Shared(Option<&'static str>),
    /// `&mut`, possibly with a lifetime
    Exclusive(Option<&'static str>),
    /// `*const`
    #[expect(dead_code, reason = "our support of raw pointers is sloppy")]
    RawConst,
    /// `*mut`
    RawMut,
}

/// A type that is a scalar both in Rust terms and shader terms.
///
/// These are a subset of [`Type`] because they appear in particular situations like atomics.
#[derive(Clone, Copy)]
pub(crate) enum Scalar {
    Bool,
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
}

impl TryFrom<naga::Scalar> for Scalar {
    type Error = UnsupportedScalarError;

    fn try_from(scalar: naga::Scalar) -> Result<Self, Self::Error> {
        let naga::Scalar { kind, width } = scalar;
        Ok(match (kind, width) {
            (naga::ScalarKind::Bool, _) => Scalar::Bool,
            (naga::ScalarKind::Sint, 4) => Scalar::I32,
            (naga::ScalarKind::Sint, 8) => Scalar::I64,
            (naga::ScalarKind::Uint, 4) => Scalar::U32,
            (naga::ScalarKind::Uint, 8) => Scalar::U64,
            (naga::ScalarKind::Float, 4) => Scalar::F32,
            (naga::ScalarKind::Float, 8) => Scalar::F64,
            _ => return Err(UnsupportedScalarError(scalar)),
        })
    }
}
pub(crate) struct UnsupportedScalarError(naga::Scalar);
impl From<UnsupportedScalarError> for crate::Error {
    fn from(error: UnsupportedScalarError) -> Self {
        // TODO: don't use Debug formatting

        crate::Error::Unimplemented(format!("no Rust translation for a scalar {:?}", error.0))
    }
}

/// Generic types defined in the runtime crate.
#[derive(Clone)]
pub(crate) enum RtGen {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Mat2x2,
    Mat2x3,
    Mat2x4,
    Mat3x2,
    Mat3x3,
    Mat3x4,
    Mat4x2,
    Mat4x3,
    Mat4x4,
}

impl RtGen {
    pub fn vector(size: VectorSize) -> Self {
        match size {
            VectorSize::Bi => RtGen::Vec2,
            VectorSize::Tri => RtGen::Vec3,
            VectorSize::Quad => RtGen::Vec4,
        }
    }

    pub fn matrix(columns: VectorSize, rows: VectorSize) -> Self {
        match (columns, rows) {
            (VectorSize::Bi, VectorSize::Bi) => RtGen::Mat2x2,
            (VectorSize::Bi, VectorSize::Tri) => RtGen::Mat2x3,
            (VectorSize::Bi, VectorSize::Quad) => RtGen::Mat2x4,
            (VectorSize::Tri, VectorSize::Bi) => RtGen::Mat3x2,
            (VectorSize::Tri, VectorSize::Tri) => RtGen::Mat3x3,
            (VectorSize::Tri, VectorSize::Quad) => RtGen::Mat3x4,
            (VectorSize::Quad, VectorSize::Bi) => RtGen::Mat4x2,
            (VectorSize::Quad, VectorSize::Tri) => RtGen::Mat4x3,
            (VectorSize::Quad, VectorSize::Quad) => RtGen::Mat4x4,
        }
    }
}

/// A Rust expression.
pub(crate) enum Expr {
    LitF16(half::f16),
    LitF32(f32),
    LitF64(f64),
    LitI16(i16),
    LitI32(i32),
    LitI64(i64),
    LitU16(u16),
    LitU32(u32),
    LitU64(u64),
    LitUsize(u32), // field is not usize because the code generator ≠ the target
    LitBool(bool),

    /// Variable, constant, function name, etc defined by the shader.
    Ident(String),

    /// The `self` token, which is not technically an identifier.
    Self_,

    /// Relative path of a function, type constructor, or constant in the runtime crate.
    RtItem(RtItem),

    /// A qualified path (without trait): `<Type>::assoc_item_path`
    QualifiedPath(Type, &'static str),

    /// `a(b)`
    Call(Box<Expr>, Vec<Expr>),
    /// `a.m(b)`
    ///
    /// Caution: Method calls should only be used for inherent methods, because trait methods could
    /// be ambiguous with other traits imported by the user.
    //---
    // TODO: make the method name an enum to centralize knowledge of what methods we use.
    Method(Box<Expr>, Cow<'static, str>, Vec<Expr>),
    /// `a[b]`
    Index(Box<Expr>, Box<Expr>),
    /// `a.0`
    TupleField(Box<Expr>, u32),
    /// `a.field_name`
    NamedField(Box<Expr>, String),

    /// `(&(|mut |raw mut |raw const) expr)`
    Borrow(PtrKind, Box<Expr>),
    /// `(*expr)`
    Deref(Box<Expr>),

    Negate(Box<Expr>),
    Not(Box<Expr>),

    // TODO: stop using naga::BinaryOperator as while it is not intrinsically wrong, it is a
    // hazard to correctness (might copy an op without thinking about whether it means the same
    // thing) and to maintenance (what if in a future version, naga::BinaryOperator gains a
    // variant that is not also a Rust operator?)
    BinOp(Box<Expr>, naga::BinaryOperator, Box<Expr>),

    #[expect(dead_code, reason = "TODO: delete if this doesn't get used")]
    As(Box<Expr>, Type),

    Array(Vec<Expr>),
    Struct(Type, Vec<(String, Expr)>),

    /// A macro call like `format!` or `panic!`: `a!("b")`
    FormatLikeMacro(&'static str, String),
}

impl Expr {
    pub fn call_rt(rt_item: RtItem, args: impl IntoIterator<Item = Expr>) -> Self {
        Expr::Call(Box::new(Expr::RtItem(rt_item)), Vec::from_iter(args))
    }

    /// Generate a call to a method of `self` or a free function, depending on `is_method`.
    pub fn call_maybe_self(
        is_method: bool,
        name: impl Into<Cow<'static, str>>,
        args: impl IntoIterator<Item = Expr>,
    ) -> Self {
        let name = name.into();
        if is_method {
            Expr::Method(Box::new(Expr::Self_), name, Vec::from_iter(args))
        } else {
            Expr::Call(
                Box::new(Expr::Ident(name.into_owned())),
                Vec::from_iter(args),
            )
        }
    }
}

/// A Rust pattern.
pub(crate) enum Pattern {
    LitI32(i32),
    LitU32(u32),
    /// `_` pattern
    Wildcard,
    /// Variable name pattern, e.g. `x`
    Binding(String),
    BindingMut(String),
}

/// An item in the value namespace (function, constant, type constructor) from the runtime crate.
pub(crate) enum RtItem {
    Scalar,
    ScalarIntoArrayIndex,
    ScalarIntoBranchCondition,
    ScalarIntoInner,
    SplatFromScalar(VectorSize),
    TextureLoad,
    TextureDimensions,
    TextureNzToScalar,
    TextureNumLevels,
    TextureNumLayers,
    TextureNumSamples,
    DiscardFn,
    IntoFn,
    ZeroFn,
}

/// A block, `{ ... statement; ... tail_expr }`.
pub(crate) struct Block(pub Vec<Statement>, pub Option<Expr>);

impl Block {
    pub fn expr(expression: Expr) -> Block {
        Block(vec![], Some(expression))
    }

    /// Returns whether this block is `{}`.
    pub fn is_empty(&self) -> bool {
        matches!(self, Self(statements, None) if statements.is_empty())
    }
}

impl FromIterator<Statement> for Block {
    fn from_iter<T: IntoIterator<Item = Statement>>(iter: T) -> Self {
        Block(Vec::from_iter(iter), None)
    }
}

/// Rust statements.
///
/// Note that this AST node contains several things that, in Rust, are actually expressions
/// that appear inside expression statements.
/// This is because it doesn’t matter for our purposes.
pub(crate) enum Statement {
    Expr(Expr),
    Let(Pattern, Option<Type>, Option<Expr>),
    Assign(Expr, Expr),
    Return(Option<Expr>),
    Break(Option<&'static str>),

    Block(Option<&'static str>, Block),
    Loop(&'static str, Block),
    If(Box<Expr>, Block, Block),
    Match(Box<Expr>, Vec<Arm>),

    /// Not actually any Rust syntax, but inserts a blank line for formatting.
    BlankLine,
}

/// A `match` arm.
pub(crate) struct Arm {
    pub pattern_alternatives: Vec<Pattern>,
    pub body: Block,
}

/// Rust attributes.
///
/// Some of these are functional (e.g. `allow()`s) and some of them have no effect and
/// exist solely for documentation purposes.
/// Arguably, the latter could be comments instead.
pub(crate) enum Attribute {
    Doc(String),

    /// `allow` attribute for ignoring lints that might occur in a generated function body.
    AllowFunctionBody,

    /// `allow(non_upper_case_globals)`
    AllowNonUpperCaseGlobals,

    /// `derive(...)`
    Derive(Cow<'static, [Trait]>),

    /// `repr(C)`
    ReprC,

    /// Entry point function’s stage. Ignored.
    Stage(naga::ShaderStage),
    /// Compute entry point function’s workgroup size. Ignored.
    WorkGroupSize([u32; 3]),
}

#[derive(Clone, Copy)]
pub(crate) enum Visibility {
    Private,
    Public,
}

/// A Rust item.
///
/// Items are those things which can appear inside a module or inside a function.
pub(crate) enum Item {
    Function(FunctionItem),
    Const(ConstItem),
    Struct(StructItem),
    Impl(Generics, Option<Trait>, Type, Vec<Item>),
}

/// A Rust function item: `fn name(...) {...}`
pub(crate) struct FunctionItem {
    pub attributes: Vec<Attribute>,
    pub visibility: Visibility,
    pub const_: bool,
    pub name: String,
    pub self_param: Option<PtrKind>,
    /// Excludes `self` parameter, if any.
    pub parameters: Vec<(Pattern, Type)>,
    pub return_type: Type,
    pub body: Block,
}

/// `const NAME = value;`
pub(crate) struct ConstItem {
    pub attributes: Vec<Attribute>,
    pub visibility: Visibility,
    pub name: String,
    pub ty: Type,
    pub value: Expr,
}

/// `struct Name {...}`
pub(crate) struct StructItem {
    pub attributes: Vec<Attribute>,
    pub visibility: Visibility,
    pub name: String,
    pub generics: Generics,
    pub fields: Vec<Field>,
}

/// One field of a `struct` declaration.
pub(crate) struct Field {
    pub attributes: Vec<Attribute>,
    pub visibility: Visibility,
    pub name: String,
    pub ty: Type,
}

// -------------------------------------------------------------------------------------------------

/// Parameters for printing a Rust AST using [`PrintAst`].
#[derive(Clone, Copy)]
pub(crate) struct PrintCtx<'ctx> {
    pub config: &'ctx crate::Config,
    pub indent: back::Level,
}

impl PrintCtx<'_> {
    fn next_indent(self) -> Self {
        Self {
            config: self.config,
            indent: self.indent.next(),
        }
    }
}

/// Trait for printing Rust AST types that require context such as indentation level.
/// Types that don’t implement this trait implement [`fmt::Display`] instead.
pub(crate) trait PrintAst {
    fn write(&self, out: &mut dyn fmt::Write, ctx: PrintCtx<'_>) -> fmt::Result;
}

impl fmt::Display for Generics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Generics::None => Ok(()),
            Generics::LtG => f.write_str("<'g>"),
        }
    }
}

impl PrintAst for Type {
    fn write(&self, out: &mut dyn fmt::Write, ctx: PrintCtx<'_>) -> fmt::Result {
        let runtime_path = &ctx.config.runtime_path;
        match self {
            Type::Unit => out.write_str("()"),
            Type::Self_ => out.write_str("Self"),
            Type::RtGen(rt_gen, scalar) => write!(out, "{runtime_path}::{rt_gen}<{scalar}>"),
            Type::Atomic(scalar) => write!(
                out,
                "::core::sync::atomic::{}",
                match scalar {
                    Scalar::Bool => "AtomicBool",
                    Scalar::I32 => "AtomicI32",
                    Scalar::I64 => "AtomicI64",
                    Scalar::U32 => "AtomicU32",
                    Scalar::U64 => "AtomicU64",
                    // TODO: should accept a narrower scalar enum instead of failing here
                    Scalar::F32 => unimplemented!("AtomicF32"),
                    Scalar::F64 => unimplemented!("AtomicF64"),
                }
            ),
            Type::BareScalar(scalar) => write!(out, "{scalar}"),
            &Type::Texture {
                dim,
                multisampled,
                arrayed,
                ref storage_type,
            } => {
                write!(
                    out,
                    "{runtime_path}::texture::Texture{multi_string}{dim_string}{array_string}<",
                    multi_string = if multisampled { "Multisampled" } else { "" },
                    dim_string = match dim {
                        naga::ImageDimension::D1 => "1d",
                        naga::ImageDimension::D2 => "2d",
                        naga::ImageDimension::D3 => "3d",
                        naga::ImageDimension::Cube => "Cube",
                    },
                    array_string = if arrayed { "Array" } else { "" },
                )?;
                storage_type.write(out, ctx)?;
                out.write_str(">")
            }
            &Type::DynTextureRead { dim, scalar } => {
                let vec = match dim {
                    naga::ImageDimension::D1 => "Scalar",
                    naga::ImageDimension::D2 => "Vec2",
                    naga::ImageDimension::D3 => "Vec3",
                    naga::ImageDimension::Cube => "Vec3",
                };
                write!(
                    out,
                    "dyn {runtime_path}::texture::Read<\
                        Coordinates = {runtime_path}::{vec}<i32>, \
                        Component = {scalar}\
                    >",
                )
            }
            Type::Sampler => write!(out, "{runtime_path}::Sampler"),
            Type::SamplerComparison => write!(out, "{runtime_path}::SamplerComparison"),
            Type::Array(element, len) => {
                out.write_str("[")?;
                element.write(out, ctx)?;
                write!(out, "; {len}]")
            }
            Type::Slice(element) => {
                out.write_str("[")?;
                element.write(out, ctx)?;
                out.write_str("]")
            }
            Type::Ptr(ptr_kind, pointee) => {
                out.write_str("&")?;
                match ptr_kind {
                    PtrKind::Shared(None) => {}
                    PtrKind::Shared(Some(lt)) => write!(out, "'{lt} ")?,
                    PtrKind::Exclusive(None) => out.write_str("mut ")?,
                    PtrKind::Exclusive(Some(lt)) => write!(out, "'{lt} mut ")?,
                    PtrKind::RawConst => out.write_str("raw const ")?,
                    PtrKind::RawMut => out.write_str("raw mut ")?,
                }
                pointee.write(out, ctx)
            }
            Type::User(name, generics) => write!(out, "{name}{generics}"),
            Type::ImplInto(inner) => {
                write!(out, "impl {runtime_path}::Into<")?;
                inner.write(out, ctx)?;
                out.write_str(">")
            }
        }
    }
}

impl PrintAst for Trait {
    fn write(&self, out: &mut dyn fmt::Write, ctx: PrintCtx<'_>) -> fmt::Result {
        let local_name = match self {
            Trait::Clone => "Clone",
            Trait::Copy => "Copy",
            Trait::Debug => "Debug",
            Trait::Default => "Default",
            Trait::PartialEq => "PartialEq",
            Trait::User(name) => {
                // Write the name literally with no path prefix.
                return out.write_str(name);
            }
        };
        let runtime_path = &ctx.config.runtime_path;
        write!(out, "{runtime_path}::{local_name}")
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Scalar::Bool => "bool",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
            Scalar::I32 => "i32",
            Scalar::I64 => "i64",
            Scalar::U32 => "u32",
            Scalar::U64 => "u64",
        })
    }
}

impl fmt::Display for RtGen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match *self {
            RtGen::Scalar => "Scalar",
            RtGen::Vec2 => "Vec2",
            RtGen::Vec3 => "Vec3",
            RtGen::Vec4 => "Vec4",
            RtGen::Mat2x2 => "Mat2x2",
            RtGen::Mat2x3 => "Mat2x3",
            RtGen::Mat2x4 => "Mat2x4",
            RtGen::Mat3x2 => "Mat3x2",
            RtGen::Mat3x3 => "Mat3x3",
            RtGen::Mat3x4 => "Mat3x4",
            RtGen::Mat4x2 => "Mat4x2",
            RtGen::Mat4x3 => "Mat4x3",
            RtGen::Mat4x4 => "Mat4x4",
        })
    }
}

impl PrintAst for Expr {
    /// Writes an expression, without any trailing newline.
    /// The expression will always be parenthesized if it could otherwise be altered by precedence
    /// (e.g. a dereference expression will be printed as `(*ptr)`, not `*ptr`, because
    /// `*ptr[index]` would otherwise be parsed as `*(ptr[index])` instead of `(*ptr)[index]`).
    fn write(&self, out: &mut dyn fmt::Write, ctx: PrintCtx<'_>) -> fmt::Result {
        let runtime_path = &ctx.config.runtime_path;
        match self {
            Expr::LitF16(value) => write!(out, "{value}f16"),
            Expr::LitF32(value) => write!(out, "{value}f32"),
            Expr::LitF64(value) => write!(out, "{value}f64"),
            Expr::LitI16(value) => write!(out, "{value}i16"),
            Expr::LitI32(value) => write!(out, "{value}i32"),
            Expr::LitI64(value) => write!(out, "{value}i64"),
            Expr::LitU16(value) => write!(out, "{value}u16"),
            Expr::LitU32(value) => write!(out, "{value}u32"),
            Expr::LitU64(value) => write!(out, "{value}u64"),
            Expr::LitUsize(value) => write!(out, "{value}usize"),
            Expr::LitBool(value) => write!(out, "{value}"),

            Expr::Ident(name) => out.write_str(name),
            Expr::Self_ => out.write_str("self"),

            Expr::QualifiedPath(ty, assoc_item) => {
                out.write_str("<")?;
                ty.write(out, ctx)?;
                write!(out, ">::{assoc_item}")
            }

            Expr::RtItem(rt_item) => write!(out, "{runtime_path}::{rt_item}"),

            Expr::Call(callee, args) => {
                callee.write(out, ctx)?;
                out.write_str("(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        out.write_str(", ")?;
                    }
                    arg.write(out, ctx)?;
                }
                out.write_str(")")
            }
            Expr::Method(receiver, method, args) => {
                receiver.write(out, ctx)?;
                write!(out, ".{method}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        out.write_str(", ")?;
                    }
                    arg.write(out, ctx)?;
                }
                out.write_str(")")
            }
            Expr::Index(collection, index) => {
                collection.write(out, ctx)?;
                out.write_str("[")?;
                index.write(out, ctx)?;
                out.write_str("]")
            }
            Expr::TupleField(tuple, field) => {
                // Parentheses are not needed because there are no prefix or infix operators that
                // have higher precedence than postfix operators.
                tuple.write(out, ctx)?;
                write!(out, ".{field}")
            }
            Expr::NamedField(tuple, field) => {
                // Parentheses are not needed because there are no prefix or infix operators that
                // have higher precedence than postfix operators.
                tuple.write(out, ctx)?;
                write!(out, ".{field}")
            }
            Expr::Borrow(ptr_kind, pointee) => {
                out.write_str("(&")?;
                // TODO: instead of ignoring lifetime name here, reject it from the type
                match ptr_kind {
                    PtrKind::Shared(_) => {}
                    PtrKind::Exclusive(_) => out.write_str("mut ")?,
                    PtrKind::RawConst => out.write_str("raw const ")?,
                    PtrKind::RawMut => out.write_str("raw mut ")?,
                }
                pointee.write(out, ctx)?;
                out.write_str(")")
            }

            Expr::Deref(pointee) => {
                out.write_str("(*")?;
                pointee.write(out, ctx)?;
                out.write_str(")")
            }
            Expr::Negate(expr) => {
                out.write_str("(-")?;
                expr.write(out, ctx)?;
                out.write_str(")")
            }
            Expr::Not(expr) => {
                out.write_str("(!")?;
                expr.write(out, ctx)?;
                out.write_str(")")
            }

            Expr::BinOp(left, op, right) => {
                out.write_str("(")?;
                left.write(out, ctx)?;
                write!(out, " {} ", back::binary_operation_str(*op))?;
                right.write(out, ctx)?;
                out.write_str(")")
            }

            Expr::As(expr, ty) => {
                out.write_str("(")?;
                expr.write(out, ctx)?;
                out.write_str(" as ")?;
                ty.write(out, ctx)?;
                out.write_str(")")
            }

            Expr::Array(exprs) => {
                out.write_str("[")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        out.write_str(", ")?;
                    }
                    expr.write(out, ctx)?;
                }
                out.write_str("]")
            }
            Expr::Struct(ty, fields) => {
                // TODO: in the general case, type must be written with an expression-context
                // type path or without generics, but currently we don’t literally construct any
                // generic structs, so that doesn’t matter.
                ty.write(out, ctx)?;
                write!(out, " {{ ")?;
                for (i, (field_name, field_expr)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.write_str(", ")?;
                    }
                    if matches!(
                        field_expr,
                        Expr::Ident(field_init_ident) if field_init_ident == field_name
                    ) {
                        // field init shorthand
                        write!(out, "{field_name}")?;
                    } else {
                        write!(out, "{field_name}: ")?;
                        field_expr.write(out, ctx)?;
                    }
                }
                write!(out, " }}")
            }

            Expr::FormatLikeMacro(macro_name, format_string) => {
                write!(
                    out,
                    "{macro_name}!({escaped_string})",
                    escaped_string = proc_macro2::Literal::string(format_string)
                )
            }
        }
    }
}

impl PrintAst for Pattern {
    fn write(&self, out: &mut dyn fmt::Write, _ctx: PrintCtx<'_>) -> fmt::Result {
        match self {
            Pattern::LitI32(value) => write!(out, "{value}i32"),
            Pattern::LitU32(value) => write!(out, "{value}u32"),
            Pattern::Wildcard => out.write_str("_"),
            Pattern::Binding(name) => out.write_str(name),
            Pattern::BindingMut(name) => write!(out, "mut {name}"),
        }
    }
}

impl fmt::Display for RtItem {
    /// Formats as a path relative to the runtime crate root.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match *self {
            RtItem::Scalar => "Scalar",
            RtItem::ScalarIntoArrayIndex => "Scalar::into_array_index",
            RtItem::ScalarIntoBranchCondition => "Scalar::into_branch_condition",
            RtItem::ScalarIntoInner => "Scalar::into_inner",
            RtItem::SplatFromScalar(size) => match size {
                VectorSize::Bi => "Vec2::splat_from_scalar",
                VectorSize::Tri => "Vec3::splat_from_scalar",
                VectorSize::Quad => "Vec4::splat_from_scalar",
            },
            RtItem::TextureLoad => "texture::Read::read_texel",
            RtItem::TextureDimensions => "texture::dimensions",
            RtItem::TextureNzToScalar => "texture::non_zero_to_scalar",
            RtItem::TextureNumLevels => "texture::Query::mip_levels",
            RtItem::TextureNumLayers => "texture::Query::array_layers",
            RtItem::TextureNumSamples => "texture::Query::samples",
            RtItem::DiscardFn => "discard",
            RtItem::IntoFn => "into",
            RtItem::ZeroFn => "zero",
        })
    }
}

impl PrintAst for Block {
    /// Formats *without* leading indentation or trailing newline, in order to be incorporated
    /// into other syntax.
    fn write(&self, out: &mut dyn fmt::Write, mut ctx: PrintCtx<'_>) -> fmt::Result {
        let Block(stmts, tail_expr) = self;
        let indent = ctx.indent;
        ctx.indent = ctx.indent.next();
        writeln!(out, "{{")?;
        for stmt in stmts {
            stmt.write(out, ctx)?;
        }
        if let Some(tail_expr) = tail_expr {
            write!(out, "{indent}", indent = ctx.indent)?;
            tail_expr.write(out, ctx)?;
            writeln!(out)?;
        }
        write!(out, "{indent}}}")
    }
}

impl PrintAst for Statement {
    fn write(&self, out: &mut dyn fmt::Write, ctx: PrintCtx<'_>) -> fmt::Result {
        let indent = ctx.indent;

        // this ctx is used for anything that should be indented but *isn't* a `Block`
        // doing its own indentation
        let next_indented_ctx = ctx.next_indent();

        match self {
            Statement::Expr(expr) => {
                write!(out, "{indent}")?;
                expr.write(out, next_indented_ctx)?;
                writeln!(out, ";")
            }
            Statement::Let(pattern, ty, init_expr) => {
                write!(out, "{indent}let ")?;
                pattern.write(out, next_indented_ctx)?;
                if let Some(ty) = ty {
                    out.write_str(": ")?;
                    ty.write(out, next_indented_ctx)?;
                }
                if let Some(init_expr) = init_expr {
                    out.write_str(" = ")?;
                    init_expr.write(out, next_indented_ctx)?;
                }
                writeln!(out, ";")
            }
            Statement::Assign(place_expr, value_expr) => {
                write!(out, "{indent}")?;
                place_expr.write(out, next_indented_ctx)?;
                out.write_str(" = ")?;
                value_expr.write(out, next_indented_ctx)?;
                writeln!(out, ";")
            }
            Statement::Return(expr) => {
                write!(out, "{indent}return")?;
                if let Some(expr) = expr {
                    out.write_str(" ")?;
                    expr.write(out, next_indented_ctx)?;
                }
                writeln!(out, ";")
            }
            Statement::Break(label) => {
                write!(out, "{indent}break")?;
                if let Some(label) = label {
                    write!(out, " '{label}")?;
                }
                writeln!(out, ";")
            }
            Statement::Block(label, block) => {
                write!(out, "{indent}")?;
                if let Some(label) = label {
                    write!(out, "'{label}: ")?;
                }
                block.write(out, ctx)?;
                // `Block` doesn't have trailing newline
                out.write_str("\n")
            }
            Statement::Loop(label, block) => {
                write!(out, "{indent}'{label}: loop ")?;
                block.write(out, ctx)?;
                // `Block` doesn't have trailing newline
                out.write_str("\n")
            }
            Statement::If(cond, then_branch, else_branch) => {
                write!(out, "{indent}if ")?;
                cond.write(out, next_indented_ctx)?;
                out.write_str(" ")?;
                then_branch.write(out, ctx)?;
                if !else_branch.is_empty() {
                    out.write_str(" else ")?;
                    else_branch.write(out, ctx)?;
                }
                writeln!(out)
            }
            Statement::Match(scrutinee, arms) => {
                let arm_indent = indent.next();
                write!(out, "{indent}match ")?;
                scrutinee.write(out, next_indented_ctx)?;
                writeln!(out, " {{")?;
                for Arm {
                    pattern_alternatives,
                    body,
                } in arms
                {
                    write!(out, "{arm_indent}")?;
                    for (i, pattern) in pattern_alternatives.iter().enumerate() {
                        if i > 0 {
                            out.write_str(" | ")?;
                        }
                        pattern.write(out, next_indented_ctx)?;
                    }
                    out.write_str(" => ")?;
                    body.write(out, next_indented_ctx)?;
                    writeln!(out)?;
                }
                writeln!(out, "{indent}}}")
            }
            Statement::BlankLine => writeln!(out),
        }
    }
}

impl PrintAst for Attribute {
    /// Writes the attribute as one line without the outer `#[]` brackets.
    fn write(&self, out: &mut dyn fmt::Write, ctx: PrintCtx<'_>) -> fmt::Result {
        let runtime_path = &ctx.config.runtime_path;
        match *self {
            Attribute::Doc(ref doc_string) => {
                write!(out, "doc = {}", proc_macro2::Literal::string(doc_string))?;
            }

            Attribute::AllowFunctionBody => {
                write!(
                    out,
                    // `clippy::all` refers to all *default* clippy lints, not all clippy lints.
                    // We’re allowing all clippy categories except for restriction, which we
                    // shall assume is on purpose, and cargo, which is irrelvant.
                    "allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery{})",
                    if ctx.config.flags.contains(WriterFlags::ALLOW_UNIMPLEMENTED) {
                        // ALLOW_UNIMPLEMENTED generates `panic!()`s which will often be
                        // followed by code that is therefore unreachable.
                        ", unreachable_code"
                    } else {
                        ""
                    }
                )?;
            }
            Attribute::AllowNonUpperCaseGlobals => {
                out.write_str("allow(non_upper_case_globals)")?;
            }
            Attribute::Derive(ref macro_paths) => {
                out.write_str(runtime_path)?;
                out.write_str("::derive(")?;
                for (i, derive) in macro_paths.iter().enumerate() {
                    if i > 0 {
                        out.write_str(", ")?;
                    }
                    derive.write(out, ctx)?;
                }
                out.write_str(")")?;
            }
            Attribute::ReprC => {
                out.write_str("repr(C)")?;
            }
            Attribute::Stage(shader_stage) => {
                let stage_str = match shader_stage {
                    naga::ShaderStage::Vertex => "vertex",
                    naga::ShaderStage::Fragment => "fragment",
                    naga::ShaderStage::Compute => "compute",
                    naga::ShaderStage::Task => "task",
                    naga::ShaderStage::Mesh => "mesh",
                    naga::ShaderStage::RayGeneration => "ray_generation",
                    naga::ShaderStage::Miss => "miss",
                    naga::ShaderStage::AnyHit => "any_hit",
                    naga::ShaderStage::ClosestHit => "closest_hit",
                };
                write!(out, "{runtime_path}::{stage_str}")?;
            }
            Attribute::WorkGroupSize(size) => {
                write!(
                    out,
                    "{runtime_path}::workgroup_size({}, {}, {})",
                    size[0], size[1], size[2]
                )?;
            }
        }
        Ok(())
    }
}

impl Attribute {
    /// Write a sequence of outer attributes, one per line, with indentation.
    pub fn write_outer(
        attributes: &[Attribute],
        out: &mut dyn fmt::Write,
        ctx: PrintCtx<'_>,
    ) -> fmt::Result {
        let indent = ctx.indent;
        for attribute in attributes {
            write!(out, "{indent}#[")?;
            attribute.write(out, ctx)?;
            writeln!(out, "]")?;
        }
        Ok(())
    }

    // add write_inner() if we need it
}

impl fmt::Display for Visibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Visibility::Private => Ok(()),
            Visibility::Public => f.write_str("pub "),
        }
    }
}

impl PrintAst for Item {
    fn write(&self, out: &mut dyn fmt::Write, mut ctx: PrintCtx<'_>) -> fmt::Result {
        match self {
            Item::Function(fn_item) => fn_item.write(out, ctx),
            Item::Const(const_item) => const_item.write(out, ctx),
            Item::Struct(struct_item) => struct_item.write(out, ctx),
            Item::Impl(generics, trait_, ty, assoc_items) => {
                let indent = ctx.indent;
                ctx.indent = ctx.indent.next();

                write!(out, "{indent}impl{generics} ")?;
                if let Some(trait_) = trait_ {
                    trait_.write(out, ctx)?;
                    out.write_str(" for ")?;
                }
                ty.write(out, ctx)?;
                writeln!(out, " {{")?;
                for item in assoc_items {
                    item.write(out, ctx)?;
                }
                writeln!(out, "{indent}}}")
            }
        }
    }
}

impl PrintAst for FunctionItem {
    fn write(&self, out: &mut dyn fmt::Write, ctx: PrintCtx<'_>) -> fmt::Result {
        let Self {
            ref attributes,
            visibility,
            const_,
            ref name,
            self_param,
            ref parameters,
            ref return_type,
            ref body,
        } = *self;

        let indent = ctx.indent;
        let wrap = wrap_function_params_heuristic(parameters);
        let param_indent = ctx.indent.next();

        Attribute::write_outer(attributes, out, ctx)?;
        write!(
            out,
            "{indent}{visibility}{constness}fn {name}(",
            constness = if const_ { "const " } else { "" },
        )?;
        if wrap {
            write!(out, "\n{param_indent}")?;
        }
        match self_param {
            Some(PtrKind::Shared(None)) => out.write_str("&self")?,
            Some(PtrKind::Shared(Some(lt))) => write!(out, "&'{lt} self")?,
            Some(PtrKind::Exclusive(None)) => out.write_str("&mut self")?,
            Some(PtrKind::Exclusive(Some(lt))) => write!(out, "&'{lt} mut self")?,
            Some(PtrKind::RawConst | PtrKind::RawMut) => {
                unimplemented!("raw pointers cannot be receivers")
            }
            None => {}
        }

        for (i, (param_pattern, param_type)) in parameters.iter().enumerate() {
            if i > 0 || self_param.is_some() {
                if wrap {
                    write!(out, ",\n{param_indent}")?;
                } else {
                    out.write_str(", ")?;
                }
            }
            param_pattern.write(out, ctx)?;
            out.write_str(": ")?;
            param_type.write(out, ctx)?;
        }
        if wrap {
            write!(out, ",\n{indent}")?;
        }
        out.write_str(")")?;
        if !matches!(return_type, Type::Unit) {
            out.write_str(" -> ")?;
            return_type.write(out, ctx)?;
        }
        out.write_str(" ")?;
        body.write(out, ctx)?;
        writeln!(out)
    }
}

/// Decide whether function parameters should be one per line instead of a single line.
#[mutants::skip]
fn wrap_function_params_heuristic(parameters: &[(Pattern, Type)]) -> bool {
    parameters.len() > 4
        || parameters.len() > 1
            && parameters
                .iter()
                .any(|(_, ty)| matches!(ty, Type::ImplInto(_) | Type::RtGen(..)))
}

impl PrintAst for ConstItem {
    fn write(&self, out: &mut dyn fmt::Write, ctx: PrintCtx<'_>) -> fmt::Result {
        let indent = ctx.indent;
        let Self {
            attributes,
            visibility,
            name,
            ty,
            value,
        } = self;
        Attribute::write_outer(attributes, out, ctx)?;
        write!(out, "{indent}{visibility}const {name}: ")?;
        ty.write(out, ctx)?;
        out.write_str(" = ")?;
        value.write(out, ctx)?;
        writeln!(out, ";")
    }
}

impl PrintAst for StructItem {
    fn write(&self, out: &mut dyn fmt::Write, ctx: PrintCtx<'_>) -> fmt::Result {
        let Self {
            attributes,
            visibility,
            name,
            generics,
            fields,
        } = self;
        let indent = ctx.indent;
        let field_ctx = ctx.next_indent();

        Attribute::write_outer(attributes, out, ctx)?;
        writeln!(out, "{indent}{visibility}struct {name}{generics} {{")?;
        for Field {
            attributes: field_attributes,
            visibility: field_visibility,
            name: field_name,
            ty: field_type,
        } in fields
        {
            Attribute::write_outer(field_attributes, out, field_ctx)?;
            write!(out, "{indent}{INDENT}{field_visibility}{field_name}: ")?;
            field_type.write(out, field_ctx)?;
            writeln!(out, ",")?;
        }
        writeln!(out, "{indent}}}")
    }
}
