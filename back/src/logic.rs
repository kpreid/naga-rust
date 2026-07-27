//! Data types and algorithms for making decisions about how shader code is translated.
//!
//! This module does not perform the translation ([`crate::writer`] does), but provides
//! data types and functions which assist in performing the translation *consistently*;
//! that is, in a way which will not result in the generated code having a type mismatch.

#[cfg(doc)]
use crate::writer::Writer;

// -------------------------------------------------------------------------------------------------

/// The Rust form that [`Writer::expr_ast_with_indirection`] should use to render a Naga
/// expression.
///
/// Sometimes a Naga `Expression` alone doesn't provide enough information to
/// choose the right rendering for it in Rust.
/// This is because the Naga IR does not have the Rust concept of “place expressions”
/// (or WGSL “references”); everything that might be read or written separately
/// from evaluating the expression itself is expressed via expressions whose Naga IR
/// types are pointers. But in Rust, we need to know whether to borrow (take a
/// reference or pointer to) a place, and if so, *how* to borrow it (`&`, `&mut`,
/// or `&raw`) to satisfy type and borrow checking.
///
/// The caller of `expr_ast_with_indirection` must therefore provide this parameter
/// to say what kind of Rust expression it wants, relative to the type of the Naga IR
/// expression.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Indirection {
    /// The Naga expression must have a pointer type, and
    /// the Rust expression will be a place expression for the referent of that pointer.
    Place,

    /// The Rust expression has the same corresponding type as the Naga expression.
    /// The Rust expression is not necessarily a mutable place; it may be borrowed
    /// immutably but not mutably.
    Ordinary,

    /// The Naga expression has a value type, but the Rust expression is a reference type.
    /// This is currently used only for texture (image) handles.
    Ref,
}

// -------------------------------------------------------------------------------------------------

/// Modifier for how scalars in Naga types are translated to Rust based on context.
///
/// In order to support translation to SIMD execution (not yet implemented as of this writing),
/// we need to convert scalars into SIMD vectors.
/// However, that conversion should apply only to things which are getting vectorized — that is,
/// function local variables, private global variables, function inputs, and function outputs —
/// but not to uniforms, struct members, workgroup variables, or the arguments of public function
/// shims.
/// This enum captures that distinction, in a way similar to [`naga::AddressSpace`] but more
/// precisely fitted to our concerns.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TypeTranslation {
    /// Scalar types are translated to standard Rust types, e.g. `[f32; 10]`.
    RustScalar,

    /// Scalar types are translated to shader-behavior types, e.g.
    /// `[rt::Scalar<f32>; 10]`, without SIMD.
    ///
    /// This is not yet implemented and currently behaves identically to `RustScalar`.
    ShaderScalar,

    /// Scalar types are translated to SIMD types which contain values for an entire workgroup.
    ///
    /// This is not yet implemented and currently behaves identically to `ShaderScalar`.
    Simd,
}
impl From<naga::AddressSpace> for TypeTranslation {
    fn from(value: naga::AddressSpace) -> Self {
        match value {
            // Everything that is stored separately per invocation gets the Simd form.
            naga::AddressSpace::Function | naga::AddressSpace::Private => Self::Simd,

            // Everything that is not stored separately, and originates from Naga, gets the
            // ShaderScalar form.
            naga::AddressSpace::Uniform
            | naga::AddressSpace::Handle
            | naga::AddressSpace::WorkGroup
            | naga::AddressSpace::Immediate
            | naga::AddressSpace::Storage { .. } => Self::ShaderScalar,
            naga::AddressSpace::TaskPayload
            | naga::AddressSpace::RayPayload
            | naga::AddressSpace::IncomingRayPayload => {
                unimplemented!("mesh and raytracing shaders are not supported")
            }
        }
    }
}
