use alloc::{
    string::{String, ToString},
    vec,
};
use arrayvec::ArrayVec;
use core::fmt::Write;

use naga::{
    Expression, Handle, Module, Scalar, ShaderStage, TypeInner,
    back::{self, INDENT},
    proc::{self, NameKey},
    valid::ModuleInfo,
};

use crate::conv::{self, BinOpClassified, unwrap_to_rust};
use crate::util::{Gensym, LevelNext};
use crate::{Config, Error};
use crate::{config::WriterFlags, util::GlobalKind};

// -------------------------------------------------------------------------------------------------

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

/// Rust attributes that we generate to correspond to shader properties that don’t
/// map directly to Rust code generation.
///
/// Currently, many of these attributes have no effect (they are discarded by the
/// corresponding attribute macros) and exist solely for documentation purposes.
/// Arguably, they could be comments instead.
enum Attribute {
    /// `allow` attribute for ignoring lints that might occur in a generated function body.
    AllowFunctionBody,

    /// Entry point function’s stage. Ignored.
    Stage(ShaderStage),
    /// Compute entry point function’s workgroup size. Ignored.
    WorkGroupSize([u32; 3]),
}

/// The Rust form that `write_expr_with_indirection` should use to render a Naga
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
/// The caller of `write_expr_with_indirection` must therefore provide this parameter
/// to say what kind of Rust expression it wants, relative to the type of the Naga IR
/// expression.
#[derive(Clone, Copy, Debug)]
enum Indirection {
    /// The Naga expression must have a pointer type, and
    /// the Rust expression will be a place expression for the referent of that pointer.
    Place,

    /// The Rust expression has the same corresponding type as the Naga expression.
    /// The Rust expression is not necessarily a mutable place; it may be borrowed
    /// immutably but not mutably.
    Ordinary,
}

/// Modifier for how scalars in Naga types are translated to Rust based on context.
///
/// In order to support translation to SIMD execution (not yet implemented as of this writing),
/// we need to convert scalars into SIMD vectors.
/// However, that conversion should apply only to things which are getting vectorized — that is,
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

/// Reserved prefix for the functions that use types chosen for the convenience of
/// execution of the shader rather than for convenient public API.
const FN_INTERNAL_TYPES_PREFIX: &str = "v_";

// -------------------------------------------------------------------------------------------------

/// [`naga`] backend allowing you to translate shader code in any language supported by Naga
/// to Rust code.
///
/// A `Writer` stores a [`Config`] and data structures that can be reused for writing multiple
/// modules.
#[allow(missing_debug_implementations, reason = "TODO")]
pub struct Writer {
    config: Config,
    names: naga::FastHashMap<NameKey, String>,
    namer: proc::Namer,
    named_expressions: naga::FastIndexMap<Handle<Expression>, String>,
}

enum ExpressionCtx<'a> {
    Global {
        module: &'a Module,
        module_info: &'a ModuleInfo,
        expressions: &'a naga::Arena<Expression>,
    },
    Function {
        module: &'a Module,
        //module_info: &'a ModuleInfo,
        func_ctx: &'a back::FunctionCtx<'a>,
    },
}

impl<'a> ExpressionCtx<'a> {
    #[track_caller]
    fn expect_func_ctx(&self) -> &'a back::FunctionCtx<'a> {
        match self {
            ExpressionCtx::Function { func_ctx, .. } => func_ctx,
            ExpressionCtx::Global { .. } => {
                unreachable!("attempting to access the function context outside of a function")
            }
        }
    }

    fn module(&self) -> &'a Module {
        match self {
            ExpressionCtx::Global { module, .. } => module,
            ExpressionCtx::Function { module, .. } => module,
        }
    }

    fn expressions(&self) -> &'a naga::Arena<Expression> {
        match self {
            ExpressionCtx::Global { expressions, .. } => expressions,
            ExpressionCtx::Function { func_ctx, .. } => func_ctx.expressions,
        }
    }

    fn types(&self) -> &'a naga::UniqueArena<naga::Type> {
        &self.module().types
    }

    fn resolve_type(&self, handle: Handle<Expression>) -> &'a TypeInner {
        match self {
            ExpressionCtx::Global { module_info, .. } => &module_info[handle],
            ExpressionCtx::Function { func_ctx, .. } => &func_ctx.info[handle].ty,
        }
        .inner_with(self.types())
    }
}

impl Writer {
    /// Creates a new [`Writer`].
    #[must_use]
    pub fn new(config: Config) -> Self {
        Writer {
            config,
            names: naga::FastHashMap::default(),
            namer: proc::Namer::default(),
            named_expressions: naga::FastIndexMap::default(),
        }
    }

    fn reset(&mut self, module: &Module) {
        let Self {
            config,
            names,
            namer,
            named_expressions,
        } = self;
        names.clear();
        namer.reset(
            module,
            conv::keywords_2024(),
            proc::KeywordSet::empty(),
            proc::CaseInsensitiveKeywordSet::empty(),
            &[FN_INTERNAL_TYPES_PREFIX],
            &mut self.names,
        );

        // TODO: We actually want to say “treat this as reserved but do not rename it”,
        // but Namer doesn’t have that option
        if let Some(g) = &config.global_struct {
            namer.call(g);
        }
        if let Some(r) = &config.resource_struct {
            namer.call(r);
        }
        named_expressions.clear();
    }

    /// Converts `module` to a string of Rust code.
    ///
    /// This function’s behavior is independent of prior uses of this [`Writer`].
    ///
    /// # Errors
    ///
    /// Returns an error if the module cannot be represented as Rust
    /// or if `out` returns an error.
    #[expect(clippy::missing_panics_doc, reason = "TODO: unfinished")]
    pub fn write(
        &mut self,
        out: &mut dyn Write,
        module: &Module,
        info: &ModuleInfo,
    ) -> BackendResult {
        if !module.overrides.is_empty() {
            self.write_unimplemented_stmt(out, "pipeline constants")?;
        }

        self.reset(module);

        // Write all structs
        for (handle, ty) in module.types.iter() {
            if let TypeInner::Struct { ref members, .. } = ty.inner {
                {
                    self.write_struct_definition(out, module, handle, members)?;
                    writeln!(out)?;
                }
            }
        }

        // Write all named constants
        let mut constants = module
            .constants
            .iter()
            .filter(|&(_, c)| c.name.is_some())
            .peekable();
        while let Some((handle, _)) = constants.next() {
            self.write_global_constant(out, module, info, handle)?;
            // Add extra newline for readability on last iteration
            if constants.peek().is_none() {
                writeln!(out)?;
            }
        }

        // If we are using resources, write the `struct` that contains them.
        let any_resource_requires_lifetime = GlobalKind::Resource
            .filter(&module.global_variables)
            .any(|(_, global)| matches!(module.types[global.ty].inner, TypeInner::Image { .. }));
        let resource_lifetime_generics = if any_resource_requires_lifetime {
            "<'g>"
        } else {
            ""
        };
        {
            let mut resource_iter = GlobalKind::Resource.filter(&module.global_variables);
            if let Some(ref resource_struct) = self.config.resource_struct {
                writeln!(
                    out,
                    "struct {resource_struct}{resource_lifetime_generics} {{"
                )?;
                for (handle, global) in resource_iter {
                    self.write_global_variable_as_struct_field(out, module, global, handle)?;
                }
                writeln!(out, "}}")?;
            } else if let Some((_, example)) = resource_iter.next() {
                return Err(Error::ResourcesNotEnabled {
                    example: example.name.clone().unwrap_or_default(),
                });
            }
        }

        // If we are using global variables, write the `struct` that contains them.
        let global_lifetime_generics =
            if self.config.global_struct.is_some() && self.config.resource_struct.is_some() {
                "<'g>"
            } else {
                // We use the resource struct like it was the global struct if we have only the
                // former, so the `impl` must use those generics.
                resource_lifetime_generics
            };
        {
            let mut global_variable_iter = GlobalKind::Variable.filter(&module.global_variables);
            if let Some(ref global_struct) = self.config.global_struct {
                let visibility = self.visibility();
                writeln!(out, "struct {global_struct}{global_lifetime_generics} {{")?;
                if let Some(ref resource_struct_name) = self.config.resource_struct {
                    writeln!(
                        out,
                        "{INDENT}{visibility}resources: \
                            &'g {resource_struct_name}{resource_lifetime_generics},"
                    )?;
                }
                for (handle, global) in global_variable_iter {
                    self.write_global_variable_as_struct_field(out, module, global, handle)?;
                }
                write!(
                    out,
                    "}}\n\
                    impl{global_lifetime_generics} {global_struct}{global_lifetime_generics} {{\n\
                    {INDENT}{visibility}const fn new(",
                )?;
                // Define new() function with parameter list depending on whether the resource
                // struct is needed.
                if let Some(ref resource_struct_name) = self.config.resource_struct {
                    write!(
                        out,
                        "resources: &'g {resource_struct_name}{resource_lifetime_generics}"
                    )?;
                }
                writeln!(out, ") -> Self {{ Self {{")?;

                if self.config.resource_struct.is_some() {
                    // Note that we reserve the name “resources” using the keyword set.
                    writeln!(out, "{INDENT}{INDENT}resources,")?;
                }
                for (handle, global) in GlobalKind::Variable.filter(&module.global_variables) {
                    self.write_global_variable_as_field_initializer(
                        out, module, info, global, handle,
                    )?;
                }
                writeln!(out, "{INDENT}}}}}\n}}")?;

                if self.config.resource_struct.is_none() {
                    // If the global struct doesn’t need a resource struct,
                    // then it can implement Default.
                    writeln!(
                        out,
                        "impl Default for {global_struct} {{ fn default() -> Self {{ Self::new() }} }}"
                    )?;
                }
            } else if let Some((_, example)) = global_variable_iter.next() {
                return Err(Error::GlobalVariablesNotEnabled {
                    example: example.name.clone().unwrap_or_default(),
                });
            }
        }

        // If we are making methods rather than free functions, start the `impl` block
        if let Some(name) = self.config.impl_type() {
            writeln!(
                out,
                "impl{global_lifetime_generics} {name}{global_lifetime_generics} {{"
            )?;
        }

        // Write all regular functions (which may or may not be in an `impl` block from above).
        for (handle, function) in module.functions.iter() {
            let fun_info = &info[handle];

            let func_ctx = back::FunctionCtx {
                ty: back::FunctionType::Function(handle),
                info: fun_info,
                expressions: &function.expressions,
                named_expressions: &function.named_expressions,
            };

            // Write the function
            self.write_function(out, module, function, &func_ctx)?;

            writeln!(out)?;
        }

        // Write all entry points
        for (index, ep) in module.entry_points.iter().enumerate() {
            let attributes = match ep.stage {
                ShaderStage::Vertex
                | ShaderStage::Fragment
                | ShaderStage::Task
                | ShaderStage::Mesh
                | ShaderStage::RayGeneration
                | ShaderStage::Miss
                | ShaderStage::AnyHit
                | ShaderStage::ClosestHit => vec![Attribute::Stage(ep.stage)],
                ShaderStage::Compute => vec![
                    Attribute::Stage(ShaderStage::Compute),
                    Attribute::WorkGroupSize(ep.workgroup_size),
                ],
            };

            self.write_attributes(out, back::Level(0), &attributes)?;

            let func_ctx = back::FunctionCtx {
                ty: back::FunctionType::EntryPoint(index.try_into().unwrap()),
                info: info.get_entry_point(index),
                expressions: &ep.function.expressions,
                named_expressions: &ep.function.named_expressions,
            };
            self.write_function(out, module, &ep.function, &func_ctx)?;

            if index < module.entry_points.len() - 1 {
                writeln!(out)?;
            }
        }

        if self.config.impl_type().is_some() {
            // End the `impl` block
            writeln!(out, "}}")?;
        }

        Ok(())
    }

    /// Writes a shader function as a pair of Rust functions.
    /// The shader function may be an entry point or not.
    /// Depending on the configuration it may be written as a method or a free function.
    fn write_function(
        &mut self,
        out: &mut dyn Write,
        module: &Module,
        func: &naga::Function,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        self.write_function_inner(out, module, func, func_ctx, true)?;
        self.write_function_inner(out, module, func, func_ctx, false)?;
        Ok(())
    }

    fn write_function_inner(
        &mut self,
        out: &mut dyn Write,
        module: &Module,
        func: &naga::Function,
        func_ctx: &back::FunctionCtx<'_>,
        is_public_shim: bool,
    ) -> BackendResult {
        let runtime_path = &self.config.runtime_path;
        let signature_type_translation = if is_public_shim {
            TypeTranslation::RustScalar
        } else {
            TypeTranslation::Simd
        };

        if !is_public_shim {
            // Don’t lint extra parentheses and such that we might emit.
            self.write_attributes(out, back::Level(0), &[Attribute::AllowFunctionBody])?;
        }

        // Write start of function item
        let func_name = match func_ctx.ty {
            back::FunctionType::EntryPoint(index) => &self.names[&NameKey::EntryPoint(index)],
            back::FunctionType::Function(handle) => &self.names[&NameKey::Function(handle)],
        };
        let name_prefix = if is_public_shim {
            ""
        } else {
            FN_INTERNAL_TYPES_PREFIX
        };
        let visibility = if is_public_shim {
            self.visibility()
        } else {
            "" // private
        };
        write!(out, "{visibility}fn {name_prefix}{func_name}(")?;

        if self.config.functions_are_methods() {
            // TODO: need to figure out whether &mut is needed
            write!(out, "&self, ")?;
        } else if func_ctx.info.global_variable_count() > 0 {
            unreachable!(
                "function has globals but globals are not enabled; \
                should have been rejected earlier"
            );
        }

        let use_into_for_arg = |arg: &naga::FunctionArgument| {
            matches!(
                module.types[arg.ty].inner,
                TypeInner::Scalar { .. } | TypeInner::Vector { .. }
            )
        };

        // Write function arguments
        for (index, arg) in func.arguments.iter().enumerate() {
            // // Write argument attribute if a binding is present
            // if let Some(ref binding) = arg.binding {
            //     self.write_attributes(&map_binding_to_attribute(binding))?;
            // }

            // Write argument name
            let argument_name = &self.names[&func_ctx.argument_key(index.try_into().unwrap())];
            write!(out, "{argument_name}: ")?;

            // Write argument type
            // TODO: When `TypeTranslation` actually does things, this and the return value
            // processing will need to be tweaked.
            if is_public_shim && use_into_for_arg(arg) {
                // Allow vectors and scalars to be converted.
                write!(out, "impl {runtime_path}::Into<")?;
                self.write_type(out, module, arg.ty, TypeTranslation::ShaderScalar)?;
                write!(out, ">")?;
            } else {
                self.write_type(out, module, arg.ty, signature_type_translation)?;
            }
            if index < func.arguments.len() - 1 {
                // Add a separator between args
                write!(out, ", ")?;
            }
        }

        write!(out, ")")?;

        // Write function return type
        if let Some(ref result) = func.result {
            write!(out, " -> ")?;
            // if let Some(ref binding) = result.binding {
            //     self.write_attributes(&map_binding_to_attribute(binding))?;
            // }
            self.write_type(out, module, result.ty, signature_type_translation)?;
        }

        write!(out, " {{")?;
        writeln!(out)?;

        if is_public_shim {
            // Write function call to the inner, internally-typed function.
            write!(out, "{INDENT}")?;
            if self.config.functions_are_methods() {
                write!(out, "self.")?;
            }
            write!(out, "{FN_INTERNAL_TYPES_PREFIX}{func_name}(")?;
            for (index, arg) in func.arguments.iter().enumerate() {
                let argument_name = &self.names[&func_ctx.argument_key(index.try_into().unwrap())];
                write!(out, "{argument_name}")?;
                if use_into_for_arg(arg) {
                    write!(out, ".into()")?;
                }
                if index < func.arguments.len() - 1 {
                    // Add a separator between args
                    write!(out, ", ")?;
                }
            }
            // The final into() converts from the internal `TypeTranslation::Simd`
            // type to the public `TypeTranslation::RustScalar` type.
            writeln!(out, ").into()")?;
        } else {
            // Write function local variables
            for (handle, local) in func.local_variables.iter() {
                // Write indentation (only for readability)
                write!(out, "{INDENT}")?;

                // Write the local name
                // The leading space is important
                write!(out, "let mut {}: ", self.names[&func_ctx.name_key(handle)])?;

                // Write the local type
                self.write_type(out, module, local.ty, TypeTranslation::Simd)?;

                // Write the local initializer if needed
                if let Some(init) = local.init {
                    write!(out, " = ")?;
                    self.write_expr(
                        out,
                        init,
                        &ExpressionCtx::Function {
                            module,
                            func_ctx,
                            //module_info: info,
                        },
                    )?;
                }

                // Finish the local with `;` and add a newline (only for readability)
                writeln!(out, ";")?;
            }

            if !func.local_variables.is_empty() {
                writeln!(out)?;
            }

            // Write the function body (statement list)
            for sta in func.body.iter() {
                // The indentation should always be 1 when writing the function body
                self.write_stmt(out, module, sta, func_ctx, back::Level(1))?;
            }

            self.named_expressions.clear();
        }
        writeln!(out, "}}")?;

        Ok(())
    }

    /// Writes one or more [`Attribute`]s as outer attributes.
    fn write_attributes(
        &self,
        out: &mut dyn Write,
        level: back::Level,
        attributes: &[Attribute],
    ) -> BackendResult {
        let runtime_path = &self.config.runtime_path;
        for attribute in attributes {
            write!(out, "{level}#[")?;
            match *attribute {
                Attribute::AllowFunctionBody => {
                    write!(
                        out,
                        // `clippy::all` refers to all *default* clippy lints, not all clippy lints.
                        // We’re allowing all clippy categories except for restriction, which we
                        // shall assume is on purpose, and cargo, which is irrelvant.
                        "allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery{})",
                        if self.config.flags.contains(WriterFlags::ALLOW_UNIMPLEMENTED) {
                            // ALLOW_UNIMPLEMENTED generates `panic!()`s which will often be
                            // followed by code that is therefore unreachable.
                            ", unreachable_code"
                        } else {
                            ""
                        }
                    )?;
                }
                Attribute::Stage(shader_stage) => {
                    let stage_str = match shader_stage {
                        ShaderStage::Vertex => "vertex",
                        ShaderStage::Fragment => "fragment",
                        ShaderStage::Compute => "compute",
                        ShaderStage::Task => "task",
                        ShaderStage::Mesh => "mesh",
                        ShaderStage::RayGeneration => "ray_generation",
                        ShaderStage::Miss => "miss",
                        ShaderStage::AnyHit => "any_hit",
                        ShaderStage::ClosestHit => "closest_hit",
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
            writeln!(out, "]")?;
        }
        Ok(())
    }

    /// Write out a definition of the struct type referred to by
    /// `handle` in `module`.
    ///
    /// Use `members` as the list of `handle`'s members. (This
    /// function is usually called after matching a `TypeInner`, so
    /// the callers already have the members at hand.)
    fn write_struct_definition(
        &self,
        out: &mut dyn Write,
        module: &Module,
        struct_handle: Handle<naga::Type>,
        members: &[naga::StructMember],
    ) -> BackendResult {
        // TODO: we will need to do custom dummy fields to ensure that vec3s have correct alignment.
        let runtime_path = &self.config.runtime_path;
        let visibility = self.visibility();
        let name: &str = &self.names[&NameKey::Type(struct_handle)];

        write!(
            out,
            "#[derive(Clone, Copy, Debug, PartialEq)]\n\
            #[repr(C)]\n\
            {visibility}struct {name}",
        )?;
        write!(out, " {{")?;
        writeln!(out)?;

        let mut dyn_sized = false;
        for (member_name, member) in self.iter_struct_members(struct_handle, members) {
            write!(out, "{INDENT}")?;

            // if let Some(ref binding) = member.binding {
            //     self.write_attributes(&map_binding_to_attribute(binding))?;
            // }

            // Write struct member name and type
            write!(out, "{visibility}{member_name}: ")?;
            self.write_type(out, module, member.ty, TypeTranslation::RustScalar)?;
            writeln!(out, ",")?;

            if module.types[member.ty]
                .inner
                .is_dynamically_sized(&module.types)
            {
                dyn_sized = true;
            }
        }

        writeln!(out, "}}")?; // end of struct item

        // Constructor (if not dynamically sized)
        if !dyn_sized {
            writeln!(out, "impl {name} {{\n{INDENT}{visibility}fn new(")?;
            // Constructor parameter list
            for (member_name, member) in self.iter_struct_members(struct_handle, members) {
                write!(
                    out,
                    "{INDENT}{INDENT}{member_name}: impl {runtime_path}::Into<"
                )?;
                self.write_type(out, module, member.ty, TypeTranslation::RustScalar)?;
                writeln!(out, ">,")?;
            }
            writeln!(out, "{INDENT}) -> Self {{ Self {{")?;
            // Struct literal
            for (member_name, _member) in self.iter_struct_members(struct_handle, members) {
                writeln!(out, "{INDENT}{INDENT}{member_name}: {member_name}.into(),")?;
            }
            writeln!(out, "{INDENT}}} }}\n}}")?;
        }

        Ok(())
    }

    fn iter_struct_members<'mem, 'name>(
        &'name self,
        struct_handle: Handle<naga::Type>,
        members: &'mem [naga::StructMember],
    ) -> impl Iterator<Item = (&'name str, &'mem naga::StructMember)> {
        members.iter().enumerate().map(move |(index, member)| {
            let name =
                &*self.names[&NameKey::StructMember(struct_handle, index.try_into().unwrap())];
            (name, member)
        })
    }

    /// Helper method used to write statements
    ///
    /// # Notes
    /// Always adds a newline
    fn write_stmt(
        &mut self,
        out: &mut dyn Write,
        module: &Module,
        stmt: &naga::Statement,
        func_ctx: &back::FunctionCtx<'_>,
        level: back::Level,
    ) -> BackendResult {
        use naga::{Expression, Statement};

        let runtime_path = &self.config.runtime_path;
        let expr_ctx = &ExpressionCtx::Function {
            module,
            func_ctx,
            //module_info: info,
        };

        match *stmt {
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let expr_info = &func_ctx.info[handle];
                    let expr_name = if let Some(name) = func_ctx.named_expressions.get(&handle) {
                        // Front end provides names for all variables at the start of writing.
                        // But we write them to step by step. We need to recache them
                        // Otherwise, we could accidentally write variable name instead of full expression.
                        // Also, we use sanitized names! It defense backend from generating variable with name from reserved keywords.
                        Some(self.namer.call(name))
                    } else {
                        let expr = &func_ctx.expressions[handle];
                        let min_ref_count = expr.bake_ref_count();
                        // Forcefully creating baking expressions in some cases to help with readability
                        let required_baking_expr = matches!(
                            *expr,
                            Expression::ImageLoad { .. }
                                | Expression::ImageQuery { .. }
                                | Expression::ImageSample { .. }
                        );
                        if min_ref_count <= expr_info.ref_count || required_baking_expr {
                            Some(Gensym(handle).to_string())
                        } else {
                            None
                        }
                    };

                    if let Some(name) = expr_name {
                        write!(out, "{level}")?;
                        self.start_named_expr(out, module, handle, func_ctx, &name)?;
                        self.write_expr(out, handle, expr_ctx)?;
                        self.named_expressions.insert(handle, name);
                        writeln!(out, ";")?;
                    }
                }
            }
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                let l2 = level.next();

                write!(
                    out,
                    "{level}if {runtime_path}::Scalar::into_branch_condition("
                )?;
                self.write_expr(out, condition, expr_ctx)?;
                writeln!(out, ") {{")?;
                for s in accept {
                    self.write_stmt(out, module, s, func_ctx, l2)?;
                }
                if !reject.is_empty() {
                    writeln!(out, "{level}}} else {{")?;
                    for s in reject {
                        self.write_stmt(out, module, s, func_ctx, l2)?;
                    }
                }
                writeln!(out, "{level}}}")?
            }
            Statement::Return { value } => {
                write!(out, "{level}return")?;
                if let Some(return_value) = value {
                    write!(out, " ")?;
                    self.write_expr(out, return_value, expr_ctx)?;
                }
                writeln!(out, ";")?;
            }
            Statement::Kill => write!(out, "{level}{runtime_path}::discard();")?,
            Statement::Store { pointer, value } => {
                self.write_store_statement(out, level, expr_ctx, pointer, value)?;
            }
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                write!(out, "{level}")?;

                // If the result is used, give it a name (`let _e10 = `).
                if let Some(expr) = result {
                    let name = Gensym(expr).to_string();
                    self.start_named_expr(out, module, expr, func_ctx, &name)?;
                    self.named_expressions.insert(expr, name);
                }

                if self.config.functions_are_methods() {
                    write!(out, "self.")?;
                }

                let func_name = &self.names[&NameKey::Function(function)];
                write!(out, "{FN_INTERNAL_TYPES_PREFIX}{func_name}(")?;
                for (index, &argument) in arguments.iter().enumerate() {
                    if index != 0 {
                        write!(out, ", ")?;
                    }
                    self.write_expr(out, argument, expr_ctx)?;
                }
                writeln!(out, ");")?
            }
            Statement::Atomic { .. } => {
                self.write_unimplemented_stmt(out, "atomic operations")?;
            }
            Statement::ImageAtomic { .. } => {
                self.write_unimplemented_stmt(out, "atomic texture operations")?;
            }
            Statement::WorkGroupUniformLoad { .. } => {
                todo!("Statement::WorkGroupUniformLoad");
            }
            Statement::ImageStore { .. } => {
                self.write_unimplemented_stmt(out, "textureStore")?;
            }
            Statement::Block(ref block) => {
                write!(out, "{level}")?;
                writeln!(out, "{{")?;
                for s in block.iter() {
                    self.write_stmt(out, module, s, func_ctx, level.next())?;
                }
                writeln!(out, "{level}}}")?;
            }
            Statement::Switch {
                selector,
                ref cases,
            } => {
                // Beginning of the match expression
                write!(out, "{level}")?;
                write!(out, "match {runtime_path}::Scalar::into_inner(")?;
                self.write_expr(out, selector, expr_ctx)?;
                writeln!(out, ") {{")?;

                // Generate each arm, collapsing empty fall-through into a single arm.
                let l2 = level.next();
                let mut new_match_arm = true;
                for case in cases {
                    if new_match_arm {
                        // Write initial indentation.
                        write!(out, "{l2}")?;
                    } else {
                        // Write or-pattern to combine cases.
                        write!(out, " | ")?;
                    }
                    // Write the case's pattern
                    match case.value {
                        naga::SwitchValue::I32(value) => {
                            write!(out, "{value}i32")?;
                        }
                        naga::SwitchValue::U32(value) => {
                            write!(out, "{value}u32")?;
                        }
                        naga::SwitchValue::Default => {
                            write!(out, "_")?;
                        }
                    }

                    new_match_arm = !(case.fall_through && case.body.is_empty());

                    // End this pattern and begin the body of this arm,
                    // if it is not fall-through.
                    if new_match_arm {
                        writeln!(out, " => {{")?;
                        for sta in case.body.iter() {
                            self.write_stmt(out, module, sta, func_ctx, l2.next())?;
                        }

                        if case.fall_through && !case.body.is_empty() {
                            // TODO
                            self.write_unimplemented_stmt(out, "switch case with fall-through")?;
                        }

                        writeln!(out, "{l2}}}")?;
                    }
                }

                writeln!(out, "{level}}}")?;
            }
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                let l2 = level.next();
                // We need a special block to implement `continuing`, and we add it unconditionally
                // so that the translation of `continue` doesn’t need to be conditional.
                writeln!(out, "{level}'naga_break: loop {{\n{l2}'naga_continue: {{")?;

                let l3 = l2.next();
                for sta in body.iter() {
                    self.write_stmt(out, module, sta, func_ctx, l3)?;
                }

                writeln!(out, "{l2}}}")?;

                for sta in continuing.iter() {
                    self.write_stmt(out, module, sta, func_ctx, l2)?;
                }

                if let Some(break_if) = break_if {
                    write!(
                        out,
                        "{l2}if {runtime_path}::Scalar::into_branch_condition(",
                        runtime_path = self.config.runtime_path,
                    )?;
                    self.write_expr(out, break_if, expr_ctx)?;
                    // No loop label needed because we are directly within the Rust `loop {}`.
                    writeln!(out, ") {{ break; }}")?;
                }

                writeln!(out, "{level}}}")?;
            }
            Statement::Break => writeln!(out, "{level}break 'naga_break;")?,
            Statement::Continue => writeln!(out, "{level}break 'naga_continue;")?,
            Statement::ControlBarrier(_) | Statement::MemoryBarrier(_) => {
                self.write_unimplemented_stmt(out, "barriers")?;
            }
            Statement::RayQuery { .. } | Statement::RayPipelineFunction(_) => {
                self.write_unimplemented_stmt(out, "raytracing")?;
            }
            Statement::SubgroupBallot { .. }
            | Statement::SubgroupCollectiveOperation { .. }
            | Statement::SubgroupGather { .. } => {
                self.write_unimplemented_stmt(out, "workgroup operations")?;
            }
            Statement::CooperativeStore { .. } => {
                self.write_unimplemented_stmt(out, "cooperative store")?;
            }
        }

        Ok(())
    }

    /// Write a statement which assigns `value_expr` to `*pointer`.
    ///
    /// This is a helper for [`Self::write_stmt()`], broken out because not all pointers will
    /// correspond to single Rust places of the correct type; sometimes we need to use setter
    /// functions, so this becomes potentially very complex.
    fn write_store_statement(
        &mut self,
        out: &mut dyn Write,
        level: back::Level,
        expr_ctx: &ExpressionCtx<'_>,
        pointer: Handle<Expression>,
        value_expr: Handle<Expression>,
    ) -> Result<(), Error> {
        let runtime_path = &self.config.runtime_path;
        let pointer_type: &TypeInner = expr_ctx.resolve_type(pointer);
        let pointer_base_type = pointer_type
            .pointer_base_type()
            .expect("Store statement’s pointer's type not a pointer type");

        // Note: `pointer_expr` is an expression that *in the Naga IR* has a pointer type,
        // but does not necessarily translate to a Rust pointer.
        let pointer_expr = &expr_ctx.expressions()[pointer];

        if let TypeInner::Atomic(_) = pointer_base_type.inner_with(expr_ctx.types()) {
            // Atomic operations are currently unsupported.
            // When they *are* supported, they will be distinct because per Rust mutability rules,
            // we don’t need to obtain a mutable place, so it will suffice to just evaluate the
            // pointer expression and call an atomic operation function on it.
            self.write_unimplemented_stmt(out, "atomic operations")?;
            return Ok(());
        }

        if let Expression::AccessIndex { base, index } = *pointer_expr {
            let access_base_type = expr_ctx.resolve_type(base);
            let access_pointer_base_type = access_base_type
                .pointer_base_type()
                .expect("Store statement’s access expression's base type not a pointer type");

            // Decide whether to use an accessor function instead of an assignment...
            if let TypeInner::Vector { .. } = access_pointer_base_type.inner_with(expr_ctx.types())
            {
                let component = back::COMPONENTS[index as usize];

                write!(out, "{level}")?;
                self.write_expr_with_indirection(out, base, expr_ctx, Indirection::Place)?;
                write!(out, ".set_{component}(")?;
                self.write_expr(out, value_expr, expr_ctx)?;
                writeln!(out, ");")?;
                return Ok(());
            }
        }

        // Fallthrough: Use Rust assignment.
        write!(out, "{level}")?;
        self.write_expr_with_indirection(out, pointer, expr_ctx, Indirection::Place)?;
        write!(out, " = ")?;

        // The fields of aggregates are (currently) translated as `TypeTranslation::RustScalar`.
        // Therefore, if we are storing to a member of a struct, we need to insert a conversion.
        // TODO: this should be factored out into a general function for converting
        // between TypeTranslations.
        match TypeTranslation::from(pointer_type.pointer_space().unwrap()) {
            TypeTranslation::RustScalar => {
                write!(out, "{runtime_path}::Scalar::into_inner(")?;
                self.write_expr(out, value_expr, expr_ctx)?;
                write!(out, ")")?;
            }
            TypeTranslation::ShaderScalar | TypeTranslation::Simd => {
                // No unwrapping
                self.write_expr(out, value_expr, expr_ctx)?;
            }
        }

        writeln!(out, ";")?;

        Ok(())
    }

    /// Return the sort of indirection that `expr`'s plain form evaluates to.
    ///
    /// An expression's 'plain form' is the shortest rendition of that
    /// expression's meaning into Rust, lacking `&` or `*` operators.
    /// Therefore, it may not have a type which matches the Naga IR expression
    /// type (because Naga does not have places, only pointers and non-pointer values).
    ///
    /// This function is in a sense a secondary return value from
    /// [`Self::write_expr_plain_form()`], but we need to have it available
    /// *before* writing the expression itself.
    fn plain_form_indirection(
        &self,
        expr: Handle<Expression>,
        expr_ctx: &ExpressionCtx<'_>,
    ) -> Indirection {
        use naga::Expression as Ex;

        // Named expressions are `let` bindings.
        // so if their type is a Naga pointer, then that must be a Rust pointer
        // as well.
        if self.named_expressions.contains_key(&expr) {
            return Indirection::Ordinary;
        }

        match expr_ctx.expressions()[expr] {
            // In Naga, a `LocalVariable(x)` expression produces a pointer,
            // but our plain form is a variable name `x`,
            // which means the caller must reference it if desired.
            Ex::LocalVariable(_) => Indirection::Place,

            // The plain form of `GlobalVariable(g)` is `self.g`, which is a
            // Rust place. However, globals in the `Handle` address space are immutable,
            // and `GlobalVariable` expressions for those produce the value directly,
            // not a pointer to it. Therefore, such expressions have `Indirection::Place`.
            // (Note that the exception for Handle is a fact about Naga IR, not this backend.)
            Ex::GlobalVariable(handle) => {
                let global = &expr_ctx.module().global_variables[handle];
                match global.space {
                    naga::AddressSpace::Handle => Indirection::Ordinary,
                    _ => Indirection::Place,
                }
            }

            // `Access` and `AccessIndex` pass through the pointer-ness of their `base` value.
            Ex::Access { base, .. } | Ex::AccessIndex { base, .. } => {
                let base_ty = expr_ctx.resolve_type(base);
                match *base_ty {
                    TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => {
                        Indirection::Place
                    }
                    _ => Indirection::Ordinary,
                }
            }
            _ => Indirection::Ordinary,
        }
    }

    fn start_named_expr(
        &self,
        out: &mut dyn Write,
        module: &Module,
        handle: Handle<Expression>,
        func_ctx: &back::FunctionCtx<'_>,
        name: &str,
    ) -> BackendResult {
        // Write variable name
        write!(out, "let {name}")?;
        if self.config.flags.contains(WriterFlags::EXPLICIT_TYPES) {
            write!(out, ": ")?;
            let ty = &func_ctx.info[handle].ty;
            // Write variable type
            match *ty {
                proc::TypeResolution::Handle(ty_handle) => {
                    self.write_type(out, module, ty_handle, TypeTranslation::Simd)?;
                }
                proc::TypeResolution::Value(ref inner) => {
                    self.write_type_inner(out, module, inner, TypeTranslation::Simd)?;
                }
            }
        }

        write!(out, " = ")?;
        Ok(())
    }

    /// Write the ordinary Rust form of `expr`.
    ///
    /// See `write_expr_with_indirection` for details.
    fn write_expr(
        &self,
        out: &mut dyn Write,
        expr: Handle<Expression>,
        expr_ctx: &ExpressionCtx<'_>,
    ) -> BackendResult {
        self.write_expr_with_indirection(out, expr, expr_ctx, Indirection::Ordinary)
    }

    /// Write `expr` as a Rust expression with the requested indirection.
    ///
    /// The expression is parenthesized if necessary to ensure it cannot be affected by precedence.
    ///
    /// This does not produce newlines or indentation.
    ///
    /// The `requested` argument indicates how the produced Rust expression’s type should relate
    /// to the Naga type of the input expression. See [`Indirection`]’s documentation for details.
    fn write_expr_with_indirection(
        &self,
        out: &mut dyn Write,
        expr: Handle<Expression>,
        expr_ctx: &ExpressionCtx<'_>,
        requested: Indirection,
    ) -> BackendResult {
        // If the plain form of the expression is not what we need, emit the
        // operator necessary to correct that.
        let plain = self.plain_form_indirection(expr, expr_ctx);
        match (requested, plain) {
            // The plain form expression will be a place.
            // Convert it to a reference to match the Naga pointer type.
            // TODO: We need to choose which borrow operator to use.
            (Indirection::Ordinary, Indirection::Place) => {
                write!(out, "(&")?;
                self.write_expr_plain_form(out, expr, expr_ctx, plain)?;
                write!(out, ")")?;
            }

            // The plain form expression will be a pointer, but the caller wants its pointee.
            // Insert a dereference operator.
            (Indirection::Place, Indirection::Ordinary) => {
                write!(out, "(*")?;
                self.write_expr_plain_form(out, expr, expr_ctx, plain)?;
                write!(out, ")")?;
            }
            // Matches.
            (Indirection::Place, Indirection::Place)
            | (Indirection::Ordinary, Indirection::Ordinary) => {
                self.write_expr_plain_form(out, expr, expr_ctx, plain)?
            }
        }

        Ok(())
    }

    /// Write the 'plain form' of `expr`.
    ///
    /// An expression's 'plain form' is the shortest rendition of that
    /// expression's meaning into Rust, lacking `&` or `*` operators.
    /// Therefore, it may not have a type which matches the Naga IR expression
    /// type (because Naga does not have places, only pointers and non-pointer values).
    ///
    /// When it does not match, this is indicated by [`Self::plain_form_indirection()`].
    /// It is the caller’s responsibility to adapt as needed, usually via
    /// [`Self::write_expr_with_indirection()`].
    ///
    /// The return type of the written expression always follows [`TypeTranslation::Simd`] form.
    /// (We will need to refine that later.)
    ///
    /// TODO: explain the indirection parameter of *this* function.
    fn write_expr_plain_form(
        &self,
        out: &mut dyn Write,
        expr: Handle<Expression>,
        expr_ctx: &ExpressionCtx<'_>,
        indirection: Indirection,
    ) -> BackendResult {
        if let Some(name) = self.named_expressions.get(&expr) {
            write!(out, "{name}")?;
            return Ok(());
        }

        let expression = &expr_ctx.expressions()[expr];
        let module = expr_ctx.module();
        let runtime_path = &self.config.runtime_path;

        match *expression {
            Expression::Literal(literal) => match literal {
                // TODO: Should we use the `half` library for f16 support
                // instead of only allowing it as a Rust unstable feature?
                naga::Literal::F16(value) => write!(out, "{runtime_path}::Scalar({value}f16)")?,
                naga::Literal::F32(value) => write!(out, "{runtime_path}::Scalar({value}f32)")?,
                naga::Literal::U32(value) => write!(out, "{runtime_path}::Scalar({value}u32)")?,
                naga::Literal::I32(value) => {
                    write!(out, "{runtime_path}::Scalar({value}i32)")?;
                }
                naga::Literal::Bool(value) => write!(out, "{runtime_path}::Scalar({value})")?,
                naga::Literal::F64(value) => write!(out, "{runtime_path}::Scalar({value}f64)")?,
                naga::Literal::I64(value) => {
                    write!(out, "{runtime_path}::Scalar({value}i64)")?;
                }
                naga::Literal::U64(value) => write!(out, "{runtime_path}::Scalar({value}u64)")?,
                naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_) => {
                    unreachable!("abstract types should not appear in IR presented to backends");
                }
            },
            Expression::Constant(handle) => {
                let constant = &module.constants[handle];
                if constant.name.is_some() {
                    write!(out, "{}", self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.write_expr(out, constant.init, expr_ctx)?;
                }
            }
            Expression::ZeroValue(ty) => {
                write!(out, "{runtime_path}::zero::<")?;
                self.write_type(out, module, ty, TypeTranslation::Simd)?;
                write!(out, ">()")?;
            }
            Expression::Compose { ty, ref components } => {
                self.write_constructor_expression(out, ty, components, expr_ctx)?;
            }
            Expression::Splat { size, value } => {
                let size = conv::vector_size_str(size);
                // TODO: emit explicit element type if explicit types requested
                write!(out, "{runtime_path}::Vec{size}::splat_from_scalar(")?;
                self.write_expr(out, value, expr_ctx)?;
                write!(out, ")")?;
            }
            Expression::Override(_) => unreachable!(),
            Expression::FunctionArgument(pos) => {
                let name_key = expr_ctx.expect_func_ctx().argument_key(pos);
                let name = &self.names[&name_key];
                write!(out, "{name}")?;
            }
            Expression::Binary { op, left, right } => match BinOpClassified::from(op) {
                BinOpClassified::Vectorizable(_) => {
                    write!(out, "(")?;
                    self.write_expr(out, left, expr_ctx)?;
                    write!(out, " {} ", back::binary_operation_str(op))?;
                    self.write_expr(out, right, expr_ctx)?;
                    write!(out, ")")?;
                }
                BinOpClassified::ScalarBool(bop) => {
                    self.write_expr(out, left, expr_ctx)?;
                    write!(out, ".{}(", bop.to_vector_method())?;
                    self.write_expr(out, right, expr_ctx)?;
                    write!(out, ")")?;
                }
                BinOpClassified::ShortCircuit(bop) => {
                    // The ".0"s are for unwrapping the input `Scalar`s
                    // TODO: when we support SIMD this will need to change completely
                    write!(out, "{runtime_path}::Scalar(")?;
                    self.write_expr(out, left, expr_ctx)?;
                    write!(out, ".0 {} ", bop.to_binary_operator())?;
                    self.write_expr(out, right, expr_ctx)?;
                    write!(out, ".0)")?;
                }
            },
            Expression::Access { base, index } => {
                self.write_expr_with_indirection(out, base, expr_ctx, indirection)?;
                // TODO: when we support SIMD, this will need to change to not be a single indexing
                // expression but a scatter/gather operation which isn’t a Rust place.
                write!(out, "[{runtime_path}::Scalar::into_array_index(")?;
                self.write_expr(out, index, expr_ctx)?;
                write!(out, ")]")?
            }
            Expression::AccessIndex { base, index } => {
                let result_ty = expr_ctx.resolve_type(expr);

                let base_ty_res = &expr_ctx.expect_func_ctx().info[base].ty;
                let mut base_ty_resolved = base_ty_res.inner_with(&module.types);

                // Find the type of container we are accessing, looking past the pointer if there
                // is one.
                let (base_container_ty_handle, base_is_pointer) = match *base_ty_resolved {
                    TypeInner::Pointer { base, space: _ } => {
                        base_ty_resolved = &module.types[base].inner;
                        (Some(base), true)
                    }
                    _ => (base_ty_res.handle(), false),
                };

                match *base_ty_resolved {
                    TypeInner::Vector { .. } => {
                        self.write_expr_with_indirection(out, base, expr_ctx, indirection)?;
                        write!(out, ".{}()", back::COMPONENTS[index as usize])?
                    }
                    TypeInner::Matrix { .. }
                    | TypeInner::Array { .. }
                    | TypeInner::BindingArray { .. }
                    | TypeInner::ValuePointer { .. } => {
                        self.write_expr_with_indirection(out, base, expr_ctx, indirection)?;
                        write!(out, "[{index}usize]")?
                    }

                    TypeInner::Struct { .. } => {
                        // TODO: This is a horrible "make the tests pass" kludge which should be
                        // replaced with more general implementation of conversion between different
                        // `TypeTranslation`s (struct contents are `TypeTranslation::RustScalar`
                        // and our result needs to be `TypeTranslation::Simd`).
                        let element_type_is_scalar = if base_is_pointer {
                            // In Naga IR, if the base is a pointer type then so is the result.
                            result_ty.pointer_base_type().is_some_and(|res| {
                                matches!(res.inner_with(&module.types), TypeInner::Scalar(_))
                            })
                        } else {
                            matches!(result_ty, TypeInner::Scalar(_))
                        };
                        if element_type_is_scalar {
                            let ty = base_container_ty_handle.unwrap();

                            write!(out, "{runtime_path}::Scalar(")?;
                            self.write_expr_with_indirection(out, base, expr_ctx, indirection)?;
                            write!(out, ".{})", &self.names[&NameKey::StructMember(ty, index)])?
                        } else {
                            // This will never panic in case the type is a `Struct`; this is not so
                            // for other types, so we can only check while inside this match arm
                            let ty = base_container_ty_handle.unwrap();

                            self.write_expr_with_indirection(out, base, expr_ctx, indirection)?;
                            write!(out, ".{}", &self.names[&NameKey::StructMember(ty, index)])?
                        }
                    }
                    ref other => unreachable!("cannot index into a {other:?}"),
                }
            }
            Expression::ImageSample { .. } => {
                self.write_unimplemented_expr(out, "texture sampling (other than textureLoad)")?;
            }
            Expression::ImageQuery { image, query } => match query {
                naga::ImageQuery::Size { level } => {
                    write!(out, "{runtime_path}::Texture::dimensions(")?;
                    self.write_expr(out, image, expr_ctx)?;
                    write!(out, ", ")?;
                    if let Some(level) = level {
                        write!(out, "{runtime_path}::Scalar::<i32>::into_inner(")?;
                        self.write_expr(out, level, expr_ctx)?;
                        write!(out, ")")?;
                    } else {
                        write!(out, "0")?;
                    }
                    write!(out, ")")?;
                }
                naga::ImageQuery::NumLevels => {
                    write!(out, "{runtime_path}::Texture::mip_levels(")?;
                    self.write_expr(out, image, expr_ctx)?;
                    write!(out, ")")?;
                }
                naga::ImageQuery::NumLayers => {
                    write!(out, "{runtime_path}::Texture::array_layers(")?;
                    self.write_expr(out, image, expr_ctx)?;
                    write!(out, ")")?;
                }
                naga::ImageQuery::NumSamples => {
                    write!(out, "{runtime_path}::Texture::samples(")?;
                    self.write_expr(out, image, expr_ctx)?;
                    write!(out, ")")?;
                }
            },
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                write!(out, "{runtime_path}::Texture::load(")?;
                self.write_expr(out, image, expr_ctx)?;
                write!(out, ", ")?;
                self.write_expr(out, coordinate, expr_ctx)?;
                write!(out, ", ")?;
                if let Some(array_index) = array_index {
                    write!(out, "{runtime_path}::Scalar::<i32>::into_inner(")?;
                    self.write_expr(out, array_index, expr_ctx)?;
                    write!(out, ")")?;
                } else {
                    write!(out, "0")?;
                }
                write!(out, ", ")?;
                if let Some(sample) = sample {
                    write!(out, "{runtime_path}::Scalar::<i32>::into_inner(")?;
                    self.write_expr(out, sample, expr_ctx)?;
                    write!(out, ")")?;
                } else {
                    write!(out, "0")?;
                }
                write!(out, ", ")?;
                if let Some(level) = level {
                    write!(out, "{runtime_path}::Scalar::<i32>::into_inner(")?;
                    self.write_expr(out, level, expr_ctx)?;
                    write!(out, ")")?;
                } else {
                    write!(out, "0")?;
                }
                write!(out, ")")?;
            }
            Expression::GlobalVariable(handle) => {
                write!(
                    out,
                    "{prefix}{name}",
                    prefix = self
                        .config
                        .global_field_access_prefix(&module.global_variables[handle]),
                    name = &self.names[&NameKey::GlobalVariable(handle)]
                )?;
            }

            Expression::As {
                expr,
                kind: to_kind,
                convert: to_width,
            } => {
                use naga::TypeInner as Ti;

                let input_type = expr_ctx.resolve_type(expr);

                self.write_expr(out, expr, expr_ctx)?;
                match (input_type, to_kind, to_width) {
                    (
                        Ti::Vector { size: _, scalar: _ } | Ti::Scalar(_),
                        to_kind,
                        Some(to_width),
                    ) => {
                        write!(
                            out,
                            ".cast_elem_as_{elem_ty}()",
                            elem_ty = unwrap_to_rust(Scalar {
                                kind: to_kind,
                                width: to_width
                            }),
                        )?;
                    }
                    _ => panic!(
                        "unimplemented cast {input_type:?} to kind {to_kind:?} width {to_width:?}"
                    ),
                }
            }
            Expression::Load { pointer } => {
                self.write_expr_with_indirection(out, pointer, expr_ctx, Indirection::Place)?;
            }
            Expression::LocalVariable(handle) => write!(
                out,
                "{}",
                self.names[&expr_ctx.expect_func_ctx().name_key(handle)]
            )?,
            Expression::ArrayLength(expr) => {
                self.write_expr(out, expr, expr_ctx)?;
                write!(out, ".len()")?;
            }

            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                self.write_expr(out, arg, expr_ctx)?;
                write!(
                    out,
                    ".{method}(",
                    method = conv::math_function_to_method(fun)
                )?;
                for arg in [arg1, arg2, arg3].into_iter().flatten() {
                    self.write_expr(out, arg, expr_ctx)?;
                    write!(out, ", ")?;
                }
                write!(out, ")")?
            }

            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                self.write_expr(out, vector, expr_ctx)?;
                write!(out, ".")?;
                for &sc in pattern[..size as usize].iter() {
                    out.write_char(back::COMPONENTS[sc as usize])?;
                }
                write!(out, "()")?;
            }
            Expression::Unary { op, expr } => {
                let unary = match op {
                    naga::UnaryOperator::Negate => "-",
                    naga::UnaryOperator::LogicalNot => "!",
                    naga::UnaryOperator::BitwiseNot => "!",
                };

                // The parentheses go on the outside because `write_expr` promises to
                // produce an unambiguous expression, so we have to wrap our expression,
                // and we don't need to wrap our own call to `write_expr`.
                write!(out, "({unary}")?;
                self.write_expr(out, expr, expr_ctx)?;

                write!(out, ")")?
            }

            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                // Calls {vector type}::select() method
                self.write_expr(out, reject, expr_ctx)?;
                write!(out, ".select(")?;
                self.write_expr(out, accept, expr_ctx)?;
                write!(out, ", ")?;
                self.write_expr(out, condition, expr_ctx)?;
                write!(out, ")")?
            }
            Expression::Derivative { .. } => {
                self.write_unimplemented_expr(out, "derivatives")?;

                // use naga::{DerivativeAxis as Axis, DerivativeControl as Ctrl};
                // let op = match (axis, ctrl) {
                //     (Axis::X, Ctrl::Coarse) => "dpdxCoarse",
                //     (Axis::X, Ctrl::Fine) => "dpdxFine",
                //     (Axis::X, Ctrl::None) => "dpdx",
                //     (Axis::Y, Ctrl::Coarse) => "dpdyCoarse",
                //     (Axis::Y, Ctrl::Fine) => "dpdyFine",
                //     (Axis::Y, Ctrl::None) => "dpdy",
                //     (Axis::Width, Ctrl::Coarse) => "fwidthCoarse",
                //     (Axis::Width, Ctrl::Fine) => "fwidthFine",
                //     (Axis::Width, Ctrl::None) => "fwidth",
                // };
                // write!(out, "{runtime_path}::{op}(")?;
                // self.write_expr(out, expr, expr_ctx)?;
                // write!(out, ")")?
            }
            Expression::Relational { fun, argument } => {
                use naga::RelationalFunction as Rf;

                match fun {
                    Rf::All => {
                        self.write_expr(out, argument, expr_ctx)?;
                        write!(out, ".all()")?
                    }
                    Rf::Any => {
                        self.write_expr(out, argument, expr_ctx)?;
                        write!(out, ".any()")?
                    }
                    Rf::IsNan => self.write_unimplemented_expr(out, "IsNan")?,
                    Rf::IsInf => self.write_unimplemented_expr(out, "IsInf")?,
                }
            }
            // Not supported yet
            Expression::RayQueryGetIntersection { .. }
            | Expression::RayQueryVertexPositions { .. }
            | Expression::CooperativeLoad { .. }
            | Expression::CooperativeMultiplyAdd { .. } => unreachable!(),
            // Nothing to do here, since call expression already cached
            Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::RayQueryProceedResult
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. } => {}
        }

        Ok(())
    }

    /// Translates [`Expression::Compose`].
    /// Examines the type to write an appropriate constructor or literal expression for it.
    ///
    /// We do not delegate to a library trait for this because the construction
    /// must be const-compatible.
    fn write_constructor_expression(
        &self,
        out: &mut dyn Write,
        ty: Handle<naga::Type>,
        components: &[Handle<Expression>],
        expr_ctx: &ExpressionCtx<'_>,
    ) -> BackendResult {
        use naga::VectorSize::{Bi, Quad, Tri};

        let ctor_name = match expr_ctx.types()[ty].inner {
            TypeInner::Vector { size, scalar: _ } => {
                // Vectors may be constructed by a collection of scalars and vectors which in
                // total have the required component count.

                let arg_sizes: ArrayVec<u8, 4> = components
                    .iter()
                    .map(|&component_expr| match *expr_ctx.resolve_type(component_expr) {
                        TypeInner::Scalar(_) => 1,
                        TypeInner::Vector { size, .. } => size as u8,
                        ref t => unreachable!(
                            "vector constructor argument should be a scalar or vector, not {t:?}"
                        ),
                    })
                    .collect();

                match (size, &*arg_sizes) {
                    (Bi, [1, 1]) => "from_scalars",
                    (Bi, [2]) => "from",
                    (Tri, [1, 1, 1]) => "from_scalars",
                    (Tri, [1, 2]) => "new_12",
                    (Tri, [2, 1]) => "new_21",
                    (Quad, [1, 1, 1, 1]) => "from_scalars",
                    (Quad, [1, 1, 2]) => "new_112",
                    (Quad, [1, 2, 1]) => "new_121",
                    (Quad, [2, 1, 1]) => "new_211",
                    (Quad, [2, 2]) => "new_22",
                    (Quad, [1, 3]) => "new_13",
                    (Quad, [3, 1]) => "new_31",
                    (Quad, [4]) => "from",
                    _ => unreachable!("vector constructor given too many components {arg_sizes:?}"),
                }
            }

            TypeInner::Array {
                base: _,
                size,
                stride: _,
            } => {
                assert!(matches!(size, naga::ArraySize::Constant(_)));

                // Write array syntax instead of a function call.
                write!(out, "[")?;
                for (index, component) in components.iter().enumerate() {
                    if index > 0 {
                        write!(out, ", ")?;
                    }
                    self.write_expr(out, *component, expr_ctx)?;
                }
                write!(out, "]")?;

                return Ok(());
            }

            // Fallback: Assume that a suitable `T::new()` associated function
            // exists.
            _ => "new",
        };

        write!(out, "<")?;
        self.write_type(out, expr_ctx.module(), ty, TypeTranslation::Simd)?;
        write!(out, ">::{ctor_name}(")?;
        for (index, component) in components.iter().enumerate() {
            if index > 0 {
                write!(out, ", ")?;
            }
            self.write_expr(out, *component, expr_ctx)?;
        }
        write!(out, ")")?;

        Ok(())
    }

    /// Write the Rust form of the Naga type `type_handle`.
    ///
    /// The form a type takes depends on the address space in which the value of that type lives.
    pub(super) fn write_type(
        &self,
        out: &mut dyn Write,
        module: &Module,
        type_handle: Handle<naga::Type>,
        type_translation: TypeTranslation,
    ) -> BackendResult {
        let ty = &module.types[type_handle];
        match ty.inner {
            TypeInner::Struct { .. } => {
                out.write_str(self.names[&NameKey::Type(type_handle)].as_str())?
            }
            ref other => self.write_type_inner(out, module, other, type_translation)?,
        }

        Ok(())
    }

    fn write_type_inner(
        &self,
        out: &mut dyn Write,
        module: &Module,
        inner: &TypeInner,
        type_translation: TypeTranslation,
    ) -> BackendResult {
        let runtime_path = &self.config.runtime_path;
        match *inner {
            TypeInner::Vector { size, scalar } => write!(
                out,
                "{runtime_path}::Vec{}<{}>",
                conv::vector_size_str(size),
                unwrap_to_rust(scalar),
            )?,
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => write!(
                out,
                "{runtime_path}::Mat{columns}x{rows}<{scalar}>",
                columns = conv::vector_size_str(columns),
                rows = conv::vector_size_str(rows),
                scalar = unwrap_to_rust(scalar),
            )?,
            TypeInner::Scalar(scalar) => match type_translation {
                TypeTranslation::RustScalar => write!(out, "{}", unwrap_to_rust(scalar))?,
                TypeTranslation::ShaderScalar | TypeTranslation::Simd => {
                    write!(out, "{runtime_path}::Scalar<{}>", unwrap_to_rust(scalar))?
                }
            },

            TypeInner::Sampler { comparison: false } => {
                write!(out, "{runtime_path}::Sampler")?;
            }
            TypeInner::Sampler { comparison: true } => {
                write!(out, "{runtime_path}::SamplerComparison")?;
            }
            TypeInner::Image {
                dim,
                arrayed: _, // TODO: support array textures
                class: _,   // TODO: might want separate traits per class
            } => {
                // TODO: we will want to support statically dispatched texture access,
                // but that will require more generics work on the resource struct.
                // `dyn` is a placeholder for further work.
                let vec = match dim {
                    naga::ImageDimension::D1 => "Scalar",
                    naga::ImageDimension::D2 => "Vec2",
                    naga::ImageDimension::D3 => "Vec3",
                    naga::ImageDimension::Cube => "Vec3",
                };
                write!(
                    out,
                    // 'g is a lifetime name which is declared on the global struct *and* the
                    // resource struct.
                    "&'g dyn {runtime_path}::Texture<\
                        Dimensions = {runtime_path}::{vec}<u32>,\
                        Coordinates = {runtime_path}::{vec}<i32>,\
                    >",
                )?;
            }
            TypeInner::Atomic(scalar) => {
                write!(
                    out,
                    "::core::sync::atomic::{}",
                    conv::atomic_type_name(scalar)?
                )?;
            }
            TypeInner::Array {
                base,
                size,
                stride: _,
            } => {
                write!(out, "[")?;
                self.write_type(out, module, base, type_translation)?;
                match size {
                    naga::ArraySize::Constant(len) => {
                        write!(out, "; {len}")?;
                    }
                    naga::ArraySize::Pending(..) => {
                        return Err(Error::Unimplemented("override array size".into()));
                    }
                    naga::ArraySize::Dynamic => {
                        // slice syntax needs no further tokens
                    }
                }
                write!(out, "]")?;
            }
            TypeInner::BindingArray { .. } => {}
            TypeInner::Pointer {
                base,
                space: pointee_space,
            } => {
                if self.config.flags.contains(WriterFlags::RAW_POINTERS) {
                    write!(out, "*mut ")?;
                } else {
                    write!(out, "&mut ")?;
                }
                self.write_type(out, module, base, TypeTranslation::from(pointee_space))?;
            }
            TypeInner::ValuePointer {
                size: _,
                scalar: _,
                space: _,
            } => {
                if self.config.flags.contains(WriterFlags::RAW_POINTERS) {
                    write!(out, "*mut ")?;
                } else {
                    write!(out, "&mut ")?;
                }
                todo!()
            }
            TypeInner::Struct { .. } => {
                unreachable!("should only see a struct by name");
            }
            TypeInner::AccelerationStructure { .. } => {
                return Err(Error::Unimplemented("type AccelerationStructure".into()));
            }
            TypeInner::RayQuery { .. } => {
                return Err(Error::Unimplemented("type RayQuery".into()));
            }
            TypeInner::CooperativeMatrix { .. } => {
                return Err(Error::Unimplemented("type CooperativeMatrix".into()));
            }
        }

        Ok(())
    }

    /// Helper method used to write global variables as translated into struct fields
    fn write_global_variable_as_struct_field(
        &self,
        out: &mut dyn Write,
        module: &Module,
        global: &naga::GlobalVariable,
        handle: Handle<naga::GlobalVariable>,
    ) -> BackendResult {
        // Write group and binding attributes if present
        let &naga::GlobalVariable {
            name: _, // renamed instead
            space,
            binding: _, // don't (yet) expose numeric binding locations
            ty,
            init: _,
            memory_decorations: _, // TODO: probably need to do things with this
        } = global;

        // Note bindings.
        // These are not emitted as attributes because Rust does not allow macro attributes to be
        // placed on struct fields.
        if let Some(naga::ResourceBinding { group, binding }) = global.binding {
            writeln!(out, "{INDENT}// group({group}) binding({binding})")?;
        }

        write!(
            out,
            "{INDENT}{visibility}{name}: ",
            visibility = self.visibility(),
            name = &self.names[&NameKey::GlobalVariable(handle)],
        )?;
        self.write_type(out, module, ty, TypeTranslation::from(space))?;
        writeln!(out, ",")?;

        Ok(())
    }
    fn write_global_variable_as_field_initializer(
        &self,
        out: &mut dyn Write,
        module: &Module,
        info: &ModuleInfo,
        global: &naga::GlobalVariable,
        handle: Handle<naga::GlobalVariable>,
    ) -> BackendResult {
        let name: &str = &self.names[&NameKey::GlobalVariable(handle)];
        write!(out, "{INDENT}{INDENT}{name}: ")?;

        if let Some(init) = global.init {
            self.write_expr(
                out,
                init,
                &ExpressionCtx::Global {
                    expressions: &module.global_expressions,
                    module,
                    module_info: info,
                },
            )?;
        } else {
            // Default will generally produce zero
            write!(out, "Default::default()")?;
        }

        // End with comma separating from the next field
        writeln!(out, ",")?;

        Ok(())
    }

    /// Writes a Rust `const` item for a [`naga::Constant`], with trailing newline.
    fn write_global_constant(
        &self,
        out: &mut dyn Write,
        module: &Module,
        info: &ModuleInfo,
        handle: Handle<naga::Constant>,
    ) -> BackendResult {
        let name = &self.names[&NameKey::Constant(handle)];
        let visibility = self.visibility();
        let init = module.constants[handle].init;

        write!(
            out,
            "#[allow(non_upper_case_globals)]\n{visibility}const {name}: "
        )?;
        self.write_type(
            out,
            module,
            module.constants[handle].ty,
            TypeTranslation::ShaderScalar,
        )?;
        write!(out, " = ")?;
        self.write_expr(
            out,
            init,
            &ExpressionCtx::Global {
                expressions: &module.global_expressions,
                module,
                module_info: info,
            },
        )?;
        writeln!(out, ";")?;

        Ok(())
    }

    /// For a feature that naga-rust does not support, either return an immediate conversion error, or emit
    /// an expression that panics when executed.
    fn write_unimplemented_expr(
        &self,
        out: &mut dyn Write,
        unimplemented_feature: &'static str,
    ) -> BackendResult {
        if self.config.flags.contains(WriterFlags::ALLOW_UNIMPLEMENTED) {
            write!(
                out,
                "unimplemented!({message})",
                message = proc_macro2::Literal::string(&alloc::format!(
                    "this shader function contains a feature which \
                    cannot yet be translated to Rust, {unimplemented_feature}"
                ))
            )?;
            Ok(())
        } else {
            Err(Error::Unimplemented(unimplemented_feature.into()))
        }
    }

    fn write_unimplemented_stmt(
        &self,
        out: &mut dyn Write,
        unimplemented_feature: &'static str,
    ) -> BackendResult {
        write!(out, "{INDENT}")?;
        self.write_unimplemented_expr(out, unimplemented_feature)?;
        writeln!(out, ";")?;
        Ok(())
    }

    fn visibility(&self) -> &'static str {
        if self.config.flags.contains(WriterFlags::PUBLIC) {
            "pub "
        } else {
            ""
        }
    }
}
