//! Data types and algorithms for making decisions about how shader code is translated.
//!
//! This module does not perform the translation ([`crate::writer`] does), but provides
//! data types and functions which assist in performing the translation *consistently*;
//! that is, in a way which will not result in the generated code having a type mismatch.

use crate::ra;

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

// -------------------------------------------------------------------------------------------------

/// Information about how a global variable declaration should be translated.
pub(crate) struct GlobalTranslation {
    pub location: GlobalLocation,

    /// Will the translation of this variable require the resource struct to borrow things?
    /// This is used for textures and storage buffers.
    pub requires_resource_struct_lifetime: bool,

    /// Should the translation of this variable have an additional indirection when put in the
    /// resource struct?
    pub declaration_indirection: Option<ra::PtrKind>,

    /// Indirection used when translating a [`naga::Expression::GlobalVariable`].
    pub global_expr_indirection: Indirection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlobalLocation {
    /// Goes in our `global_struct`.
    Variable,
    /// Goes in our `resource_struct`.
    Resource,
}

impl GlobalTranslation {
    pub fn get(module: &naga::Module, global: &naga::GlobalVariable) -> Self {
        let (declaration_indirection, global_expr_indirection, location): (
            Option<ra::PtrKind>,
            Indirection,
            GlobalLocation,
        ) = match global.space {
            // These globals are stored by value in the global_struct.
            // `GlobalVariable` expressions should produce pointers to them (references, in WGSL
            // terms), so we use `Indirection::Place` to signal that we will produce a Rust place
            // corresponding to a WGSL reference.
            naga::AddressSpace::Private => (None, Indirection::Place, GlobalLocation::Variable),
            naga::AddressSpace::WorkGroup => (None, Indirection::Place, GlobalLocation::Variable),

            naga::AddressSpace::Uniform => (None, Indirection::Place, GlobalLocation::Resource),

            // `GlobalVariable` expressions in the `Handle` address space are shallow-immutable
            // and produce the value, not a pointer to it. (This is a fact about Naga IR.)
            //  Therefore, such expressions have `Indirection::Ordinary` to copy the value
            // from the global struct.
            naga::AddressSpace::Handle => (None, Indirection::Ordinary, GlobalLocation::Resource),

            // Storage buffers are variable-length, so they are referenced in the
            // resource_struct, so they are `Ordinary` (they match the Naga IR typing).
            naga::AddressSpace::Storage { access } => (
                Some(
                    // TODO: Checking for ATOMIC isn't helping us make the decision we actually
                    // want (not to use &mut for values accessed atomically). Figure out whether
                    // we should be doing this at all, and what the right thing is.
                    if access.contains(naga::StorageAccess::STORE)
                        && !access.contains(naga::StorageAccess::ATOMIC)
                    {
                        ra::PtrKind::Exclusive(Some("g"))
                    } else {
                        ra::PtrKind::Shared(Some("g"))
                    },
                ),
                Indirection::Ref,
                GlobalLocation::Resource,
            ),

            // Not actually supported.
            naga::AddressSpace::Immediate => todo!(),
            naga::AddressSpace::TaskPayload => todo!(),
            naga::AddressSpace::RayPayload => todo!(),
            naga::AddressSpace::IncomingRayPayload => todo!(),

            // Never appears as a global.
            naga::AddressSpace::Function => unreachable!(),
        };

        let type_is_image = matches!(module.types[global.ty].inner, naga::TypeInner::Image { .. });

        let requires_resource_struct_lifetime = type_is_image || declaration_indirection.is_some();

        Self {
            location,
            requires_resource_struct_lifetime,
            declaration_indirection,
            global_expr_indirection,
        }
    }
}

impl GlobalLocation {
    /// Iterate over all global variables in `module` which have this location.
    pub fn filter(
        self,
        module: &naga::Module,
    ) -> impl Iterator<
        Item = (
            naga::Handle<naga::GlobalVariable>,
            &naga::GlobalVariable,
            GlobalTranslation,
        ),
    > {
        module
            .global_variables
            .iter()
            .filter_map(move |(handle, global)| {
                let var_translation = GlobalTranslation::get(module, global);
                if var_translation.location == self {
                    Some((handle, global, var_translation))
                } else {
                    None
                }
            })
    }
}
