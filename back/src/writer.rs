use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use arrayvec::ArrayVec;
use core::fmt::Write;

use naga::{
    Expression, Handle, Module, ShaderStage, TypeInner, back,
    proc::{self, NameKey},
    valid::ModuleInfo,
};

use crate::conv::{self, BinOpClassified};
use crate::ra::{self, PrintAst as _};
use crate::util::Gensym;
use crate::{Config, Error};
use crate::{config::WriterFlags, util::GlobalKind};

// -------------------------------------------------------------------------------------------------

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

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
enum Indirection {
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
    pub fn write(
        &mut self,
        out: &mut dyn Write,
        module: &Module,
        info: &ModuleInfo,
    ) -> BackendResult {
        let top_level_items = self.translate_module(module, info)?;

        let print_ctx = ra::PrintCtx {
            config: &self.config,
            indent: back::Level(0),
        };
        for item in top_level_items {
            item.write(out, print_ctx)?;
        }

        Ok(())
    }

    /// Converts `module` to a Rust AST.
    ///
    /// This function’s behavior is independent of prior uses of this [`Writer`].
    ///
    /// (It is not public because the AST is not public)
    pub(crate) fn translate_module(
        &mut self,
        module: &Module,
        info: &ModuleInfo,
    ) -> Result<Vec<ra::Item>, Error> {
        if !module.overrides.is_empty() {
            return Err(Error::Unimplemented("pipeline constants".into()));
        }

        self.reset(module);

        let visibility = self.visibility();
        let mut top_level_items: Vec<ra::Item> = Vec::new();

        // Translate all structs
        for (handle, ty) in module.types.iter() {
            if let TypeInner::Struct { ref members, .. } = ty.inner {
                top_level_items.extend(self.translate_struct_definition(module, handle, members)?);
            }
        }

        // Translate all named constants
        for (handle, _) in module.constants.iter().filter(|&(_, c)| c.name.is_some()) {
            top_level_items.push(ra::Item::Const(
                self.translate_global_constant(module, info, handle)?,
            ));
        }

        // If we are using resources, write the `struct` that contains them.
        let any_resource_requires_lifetime = GlobalKind::Resource
            .filter(&module.global_variables)
            .any(|(_, global)| matches!(module.types[global.ty].inner, TypeInner::Image { .. }));
        let resource_lifetime_generics = if any_resource_requires_lifetime {
            ra::Generics::LtG
        } else {
            ra::Generics::None
        };
        {
            let mut resource_iter = GlobalKind::Resource.filter(&module.global_variables);
            if let Some(ref resource_struct_name) = self.config.resource_struct {
                let resource_struct_ast = ra::StructItem {
                    attributes: vec![],
                    visibility: ra::Visibility::Private, // TODO: wrong visibility
                    name: resource_struct_name.clone(),
                    generics: resource_lifetime_generics,
                    fields: resource_iter
                        .map(|(handle, global)| {
                            self.translate_global_variable_to_struct_field(module, global, handle)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                };

                top_level_items.push(ra::Item::Struct(resource_struct_ast));
            } else if let Some((_, example)) = resource_iter.next() {
                return Err(Error::ResourcesNotEnabled {
                    example: example.name.clone().unwrap_or_default(),
                });
            }
        }

        let ref_to_resources_ty = self.config.resource_struct.as_ref().map(|name| {
            ra::Type::Ptr(
                ra::PtrKind::Shared(Some("g")),
                Box::new(ra::Type::User(name.clone(), resource_lifetime_generics)),
            )
        });

        // If we are using global variables, write the `struct` that contains them.
        let global_lifetime_generics =
            if self.config.global_struct.is_some() && self.config.resource_struct.is_some() {
                ra::Generics::LtG
            } else {
                // We use the resource struct like it was the global struct if we have only the
                // former, so the `impl` must use those generics.
                resource_lifetime_generics
            };
        {
            let mut global_variable_iter = GlobalKind::Variable.filter(&module.global_variables);
            if let Some(ref global_struct_name) = self.config.global_struct {
                let mut global_struct_fields = Vec::new();
                if let Some(ref ref_to_resources_ty) = ref_to_resources_ty {
                    global_struct_fields.push(ra::Field {
                        attributes: vec![],
                        visibility,
                        name: "resources".into(),
                        ty: ref_to_resources_ty.clone(),
                    });
                }
                for (handle, global) in global_variable_iter {
                    global_struct_fields.push(
                        self.translate_global_variable_to_struct_field(module, global, handle)?,
                    );
                }

                let global_struct_item = ra::StructItem {
                    attributes: vec![],
                    visibility: ra::Visibility::Private, // TODO: existing behavior but is it right?
                    name: global_struct_name.clone(),
                    generics: global_lifetime_generics,
                    fields: global_struct_fields,
                };
                top_level_items.push(ra::Item::Struct(global_struct_item));

                let global_struct_constructor_fn = ra::FunctionItem {
                    attributes: vec![],
                    visibility,
                    const_: true,
                    name: "new".into(),
                    // Define new() function with parameter list depending on whether the resource
                    // struct is needed.
                    parameters: if let Some(ref_to_resources_ty) = ref_to_resources_ty {
                        vec![(
                            ra::Pattern::Binding("resources".into()),
                            ref_to_resources_ty,
                        )]
                    } else {
                        vec![]
                    },
                    self_param: None,
                    return_type: ra::Type::Self_,
                    body: ra::Block::expr(ra::Expr::Struct(ra::Type::Self_, {
                        let mut fields = Vec::new();
                        if self.config.resource_struct.is_some() {
                            // Note that we reserve the name “resources” using the keyword set.
                            fields.push(("resources".into(), ra::Expr::Ident("resources".into())));
                        }
                        for (handle, global) in
                            GlobalKind::Variable.filter(&module.global_variables)
                        {
                            fields.push((
                                self.names[&NameKey::GlobalVariable(handle)].clone(),
                                self.global_variable_as_field_initializer_expr(
                                    module, info, global,
                                )?,
                            ));
                        }
                        fields
                    })),
                };

                top_level_items.push(ra::Item::Impl(
                    global_lifetime_generics,
                    None,
                    ra::Type::User(global_struct_name.clone(), global_lifetime_generics),
                    vec![ra::Item::Function(global_struct_constructor_fn)],
                ));

                if self.config.resource_struct.is_none() {
                    // If the global struct doesn’t need a resource struct,
                    // then it can implement Default.
                    top_level_items.push(ra::Item::Impl(
                        global_lifetime_generics,
                        Some(ra::Trait::Default),
                        ra::Type::User(global_struct_name.clone(), global_lifetime_generics),
                        vec![ra::Item::Function(ra::FunctionItem {
                            attributes: Vec::new(),
                            visibility: ra::Visibility::Private, // actually implicit-public
                            const_: false,
                            name: "default".into(),
                            self_param: None,
                            parameters: vec![],
                            return_type: ra::Type::Self_,
                            body: ra::Block::expr(ra::Expr::Call(
                                Box::new(ra::Expr::QualifiedPath(ra::Type::Self_, "new")),
                                vec![],
                            )),
                        })],
                    ));
                }
            } else if let Some((_, example)) = global_variable_iter.next() {
                return Err(Error::GlobalVariablesNotEnabled {
                    example: example.name.clone().unwrap_or_default(),
                });
            }
        }

        // Collects all items that go in the `impl Globals` if there is one.
        let mut maybe_impl_items: Vec<ra::Item> = Vec::new();

        // Translate all regular functions (which may or may not go in an `impl` block).
        for (handle, function) in module.functions.iter() {
            let fun_info = &info[handle];

            let func_ctx = back::FunctionCtx {
                ty: back::FunctionType::Function(handle),
                info: fun_info,
                expressions: &function.expressions,
                named_expressions: &function.named_expressions,
            };

            // Translate the function
            maybe_impl_items.extend(self.translate_function(
                module,
                function,
                &func_ctx,
                vec![],
            )?);
        }

        // Translate all entry points
        for (index, ep) in module.entry_points.iter().enumerate() {
            let entry_point_attributes = match ep.stage {
                ShaderStage::Vertex
                | ShaderStage::Fragment
                | ShaderStage::Task
                | ShaderStage::Mesh
                | ShaderStage::RayGeneration
                | ShaderStage::Miss
                | ShaderStage::AnyHit
                | ShaderStage::ClosestHit => vec![ra::Attribute::Stage(ep.stage)],
                ShaderStage::Compute => vec![
                    ra::Attribute::Stage(ShaderStage::Compute),
                    ra::Attribute::WorkGroupSize(ep.workgroup_size),
                ],
            };

            let func_ctx = back::FunctionCtx {
                ty: back::FunctionType::EntryPoint(index.try_into().unwrap()),
                info: info.get_entry_point(index),
                expressions: &ep.function.expressions,
                named_expressions: &ep.function.named_expressions,
            };
            maybe_impl_items.extend(self.translate_function(
                module,
                &ep.function,
                &func_ctx,
                entry_point_attributes,
            )?);
        }

        // If we are making methods rather than free functions, start the `impl` block
        if let Some(name) = self.config.impl_type() {
            top_level_items.push(ra::Item::Impl(
                global_lifetime_generics,
                None,
                ra::Type::User(name.to_string(), global_lifetime_generics),
                maybe_impl_items,
            ))
        } else {
            top_level_items.extend(maybe_impl_items);
        }

        Ok(top_level_items)
    }

    /// Translates a shader function to a pair of Rust functions.
    /// The shader function may be an entry point or not.
    /// Depending on the configuration it may be written as a method or a free function.
    fn translate_function(
        &mut self,
        module: &Module,
        func: &naga::Function,
        func_ctx: &back::FunctionCtx<'_>,
        extra_attributes: Vec<ra::Attribute>,
    ) -> Result<[ra::Item; 2], Error> {
        Ok([
            self.translate_function_inner(module, func, func_ctx, true, extra_attributes)?,
            self.translate_function_inner(module, func, func_ctx, false, vec![])?,
        ])
    }

    fn translate_function_inner(
        &mut self,
        module: &Module,
        func: &naga::Function,
        func_ctx: &back::FunctionCtx<'_>,
        is_public_shim: bool,
        mut attributes: Vec<ra::Attribute>,
    ) -> Result<ra::Item, Error> {
        let signature_type_translation = if is_public_shim {
            TypeTranslation::RustScalar
        } else {
            TypeTranslation::Simd
        };

        if !is_public_shim {
            // Don’t lint extra parentheses and such that we might emit.
            attributes.push(ra::Attribute::AllowFunctionBody);
        }

        let shader_func_name = match func_ctx.ty {
            back::FunctionType::EntryPoint(index) => &self.names[&NameKey::EntryPoint(index)],
            back::FunctionType::Function(handle) => &self.names[&NameKey::Function(handle)],
        };
        let name_prefix = if is_public_shim {
            ""
        } else {
            FN_INTERNAL_TYPES_PREFIX
        };
        let rust_func_name = format!("{name_prefix}{shader_func_name}");
        let visibility = if is_public_shim {
            self.visibility()
        } else {
            ra::Visibility::Private
        };

        let self_param = if self.config.functions_are_methods() {
            // TODO: need to figure out whether &mut is needed
            Some(ra::PtrKind::Shared(None))
        } else if func_ctx.info.global_variable_count() > 0 {
            unreachable!(
                "function has globals but globals are not enabled; \
                should have been rejected earlier"
            )
        } else {
            None
        };

        let use_into_for_arg = |arg: &naga::FunctionArgument| {
            matches!(
                module.types[arg.ty].inner,
                TypeInner::Scalar { .. } | TypeInner::Vector { .. }
            )
        };

        // Translate function arguments
        let mut rust_params: Vec<(ra::Pattern, ra::Type)> =
            Vec::with_capacity(func.arguments.len());
        for (index, arg) in func.arguments.iter().enumerate() {
            let argument_name = &self.names[&func_ctx.argument_key(index.try_into().unwrap())];

            // TODO: When `TypeTranslation` actually does things, this and the return value
            // processing will need to be tweaked.
            let rust_type = if is_public_shim && use_into_for_arg(arg) {
                // Allow vectors and scalars to be converted.
                ra::Type::ImplInto(Box::new(self.type_ast(
                    module,
                    arg.ty,
                    TypeTranslation::ShaderScalar,
                )?))
            } else {
                self.type_ast(module, arg.ty, signature_type_translation)?
            };

            rust_params.push((ra::Pattern::Binding(argument_name.clone()), rust_type));
        }

        let return_type = if let Some(ref result) = func.result {
            // if let Some(ref binding) = result.binding {
            //     self.write_attributes(&map_binding_to_attribute(binding))?;
            // }
            self.type_ast(module, result.ty, signature_type_translation)?
        } else {
            ra::Type::Unit
        };

        let body = if is_public_shim {
            // Translate function call to the inner, internally-typed function.

            let mut call_args = Vec::new();
            for (index, arg) in func.arguments.iter().enumerate() {
                let argument_name = &self.names[&func_ctx.argument_key(index.try_into().unwrap())];
                let argument_expr = if use_into_for_arg(arg) {
                    ra::Expr::call_rt(ra::RtItem::IntoFn, [ra::Expr::Ident(argument_name.clone())])
                } else {
                    ra::Expr::Ident(argument_name.clone())
                };
                call_args.push(argument_expr);
            }

            // The final into() converts from the internal `TypeTranslation::Simd`
            // type to the public `TypeTranslation::RustScalar` type.
            // TODO: calling into as a trait method can be ambiguous
            ra::Block::expr(ra::Expr::call_rt(
                ra::RtItem::IntoFn,
                [ra::Expr::call_maybe_self(
                    self.config.functions_are_methods(),
                    format!("{FN_INTERNAL_TYPES_PREFIX}{shader_func_name}"),
                    call_args,
                )],
            ))
        } else {
            let mut body_statements = Vec::new();

            // Define function local variables
            for (handle, local) in func.local_variables.iter() {
                let local_name = self.names[&func_ctx.name_key(handle)].clone();
                let init_expression = if let Some(init) = local.init {
                    Some(self.translate_expr(
                        init,
                        &ExpressionCtx::Function {
                            module,
                            func_ctx,
                            //module_info: info,
                        },
                    )?)
                } else {
                    None
                };

                body_statements.push(ra::Statement::Let(
                    ra::Pattern::BindingMut(local_name),
                    Some(self.type_ast(module, local.ty, TypeTranslation::Simd)?),
                    init_expression,
                ));
            }
            if !func.local_variables.is_empty() {
                body_statements.push(ra::Statement::BlankLine);
            }

            // Translate the function body (statement list)
            for sta in func.body.iter() {
                body_statements.extend(self.translate_statement(module, sta, func_ctx)?);
            }

            self.named_expressions.clear();

            ra::Block(body_statements, None)
        };

        Ok(ra::Item::Function(ra::FunctionItem {
            attributes,
            visibility,
            const_: false,
            name: rust_func_name,
            self_param,
            parameters: rust_params,
            return_type,
            body,
        }))
    }

    /// Translates the definition of the struct type referred to by `handle` in `module`.
    /// Generates a `struct` item, and a constructor function if the struct type is not
    /// dynamically sized.
    ///
    /// Use `members` as the list of `handle`'s members. (This
    /// function is usually called after matching a `TypeInner`, so
    /// the callers already have the members at hand.)
    fn translate_struct_definition(
        &self,
        module: &Module,
        struct_handle: Handle<naga::Type>,
        members: &[naga::StructMember],
    ) -> Result<Vec<ra::Item>, Error> {
        // TODO: we will need to do custom dummy fields to ensure that vec3s have correct alignment.
        let visibility = self.visibility();
        let name = &self.names[&NameKey::Type(struct_handle)];
        let struct_attributes = vec![ra::Attribute::DeriveStructTraits, ra::Attribute::ReprC];
        let self_ty = ra::Type::User(name.clone(), ra::Generics::None);

        let mut fields: Vec<ra::Field> = Vec::with_capacity(members.len());
        let mut dyn_sized = false;
        for (member_name, member) in self.iter_struct_members(struct_handle, members) {
            // TODO: add bindings as doc-comments ?
            // if let Some(ref binding) = member.binding {
            //     map_binding_to_attribute(binding);
            // }

            fields.push(ra::Field {
                attributes: vec![],
                visibility,
                name: member_name.to_string(),
                ty: self.type_ast(module, member.ty, TypeTranslation::RustScalar)?,
            });

            if module.types[member.ty]
                .inner
                .is_dynamically_sized(&module.types)
            {
                dyn_sized = true;
            }
        }

        let mut items = vec![ra::Item::Struct(ra::StructItem {
            attributes: struct_attributes,
            visibility,
            name: name.clone(),
            generics: ra::Generics::None,
            fields,
        })];

        // Constructor (if not dynamically sized)
        if !dyn_sized {
            let mut constructor_parameters = Vec::new();
            let mut constructor_fields = Vec::new();
            for (member_name, member) in self.iter_struct_members(struct_handle, members) {
                constructor_parameters.push((
                    ra::Pattern::Binding(member_name.to_string()),
                    ra::Type::ImplInto(Box::new(self.type_ast(
                        module,
                        member.ty,
                        TypeTranslation::RustScalar,
                    )?)),
                ));
                constructor_fields.push((
                    member_name.to_string(),
                    ra::Expr::call_rt(
                        ra::RtItem::IntoFn,
                        [ra::Expr::Ident(member_name.to_string())],
                    ),
                ));
            }

            let constructor_fn = ra::FunctionItem {
                attributes: vec![],
                visibility,
                const_: false,
                name: "new".into(),
                self_param: None,
                parameters: constructor_parameters,
                return_type: ra::Type::Self_,
                body: ra::Block::expr(ra::Expr::Struct(ra::Type::Self_, constructor_fields)),
            };

            items.push(ra::Item::Impl(
                ra::Generics::None,
                None,
                self_ty,
                vec![ra::Item::Function(constructor_fn)],
            ));
        }

        Ok(items)
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

    fn translate_block(
        &mut self,
        module: &Module,
        stmts: &[naga::Statement],
        func_ctx: &back::FunctionCtx<'_>,
    ) -> Result<ra::Block, Error> {
        Ok(ra::Block(
            self.translate_statements(module, stmts, func_ctx)?,
            None,
        ))
    }

    fn translate_statements(
        &mut self,
        module: &Module,
        stmts: &[naga::Statement],
        func_ctx: &back::FunctionCtx<'_>,
    ) -> Result<Vec<ra::Statement>, Error> {
        let mut output: Vec<ra::Statement> = Vec::with_capacity(stmts.len());
        for s in stmts {
            output.append(&mut self.translate_statement(module, s, func_ctx)?);
        }
        Ok(output)
    }

    fn translate_statement(
        &mut self,
        module: &Module,
        stmt: &naga::Statement,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> Result<Vec<ra::Statement>, Error> {
        use naga::{Expression, Statement};

        let expr_ctx = &ExpressionCtx::Function {
            module,
            func_ctx,
            //module_info: info,
        };

        Ok(match *stmt {
            Statement::Emit(ref range) => {
                let mut output = Vec::new();
                for handle in range.clone() {
                    let expr_info = &func_ctx.info[handle];
                    // TODO: this naming logic originated as a copy from the WGSL backend and it is
                    // unclear what the rationale is (original comments were unclear).
                    let expr_name = if let Some(name) = func_ctx.named_expressions.get(&handle) {
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
                        output.push(self.let_ast(
                            module,
                            handle,
                            func_ctx,
                            name.clone(),
                            self.translate_expr(handle, expr_ctx)?,
                        )?);
                        self.named_expressions.insert(handle, name);
                    }
                }
                output
            }
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                vec![ra::Statement::If(
                    Box::new(ra::Expr::call_rt(
                        ra::RtItem::ScalarIntoBranchCondition,
                        [self.translate_expr(condition, expr_ctx)?],
                    )),
                    self.translate_block(module, accept, func_ctx)?,
                    self.translate_block(module, reject, func_ctx)?,
                )]
            }
            Statement::Return { value } => {
                vec![ra::Statement::Return(
                    value
                        .map(|v| self.translate_expr(v, expr_ctx))
                        .transpose()?,
                )]
            }
            Statement::Kill => vec![ra::Statement::Expr(ra::Expr::call_rt(
                ra::RtItem::DiscardFn,
                [],
            ))],
            Statement::Store { pointer, value } => {
                vec![self.translate_store_statement(expr_ctx, pointer, value)?]
            }
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                let func_name = &self.names[&NameKey::Function(function)];
                let rust_func_name = format!("{FN_INTERNAL_TYPES_PREFIX}{func_name}");
                let rust_args = self.translate_exprs(expr_ctx, arguments.iter().copied())?;

                let translated_call_expr = ra::Expr::call_maybe_self(
                    self.config.functions_are_methods(),
                    Cow::Owned(rust_func_name),
                    rust_args,
                );

                // If the result is used, give it a name (`let _e10 = `).
                if let Some(result_expr) = result {
                    let name = Gensym(result_expr).to_string();
                    self.named_expressions.insert(result_expr, name.clone());
                    vec![self.let_ast(module, result_expr, func_ctx, name, translated_call_expr)?]
                } else {
                    vec![ra::Statement::Expr(translated_call_expr)]
                }
            }
            Statement::Atomic { .. } => vec![self.unimplemented_stmt("atomic operations")?],
            Statement::ImageAtomic { .. } => {
                vec![self.unimplemented_stmt("atomic texture operations")?]
            }
            Statement::WorkGroupUniformLoad { .. } => {
                todo!("Statement::WorkGroupUniformLoad");
            }
            Statement::ImageStore { .. } => vec![self.unimplemented_stmt("textureStore")?],
            Statement::Block(ref block) => {
                vec![ra::Statement::Block(
                    None, // no label
                    self.translate_block(module, block, func_ctx)?,
                )]
            }
            Statement::Switch {
                selector,
                ref cases,
            } => {
                // Generate each arm, collapsing empty fall-through into a single arm.
                let mut arms: Vec<ra::Arm> = Vec::with_capacity(cases.len());
                let mut previous_case_was_fall_through = false;
                for case in cases {
                    let &naga::SwitchCase {
                        value,
                        ref body,
                        fall_through,
                    } = case;
                    if fall_through && !body.is_empty() {
                        // TODO
                        return Ok(vec![self.unimplemented_stmt(
                            "switch case with statements and fall-through",
                        )?]);
                    }

                    let pattern = match value {
                        naga::SwitchValue::I32(value) => ra::Pattern::LitI32(value),
                        naga::SwitchValue::U32(value) => ra::Pattern::LitU32(value),
                        naga::SwitchValue::Default => ra::Pattern::Wildcard,
                    };

                    let translated_body = self.translate_block(module, body, func_ctx)?;

                    if previous_case_was_fall_through {
                        let existing_arm = arms.last_mut().unwrap();
                        existing_arm.pattern_alternatives.push(pattern);
                        existing_arm.body.0.extend(translated_body.0);
                    } else {
                        arms.push(ra::Arm {
                            pattern_alternatives: vec![pattern],
                            body: translated_body,
                        });
                    }

                    previous_case_was_fall_through = case.fall_through;
                }

                vec![ra::Statement::Match(
                    Box::new(ra::Expr::call_rt(
                        ra::RtItem::ScalarIntoInner,
                        [self.translate_expr(selector, expr_ctx)?],
                    )),
                    arms,
                )]
            }
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                let mut rust_loop_body = Vec::with_capacity(3);

                rust_loop_body.push(ra::Statement::Block(
                    Some("naga_continue"),
                    self.translate_block(module, body, func_ctx)?,
                ));

                // continuing block is the target of a Naga "continue", so it is outside the
                // labeled block `'naga_continue: { ...body... }`.
                rust_loop_body
                    .append(&mut self.translate_statements(module, continuing, func_ctx)?);

                if let Some(break_if) = break_if {
                    rust_loop_body.push(ra::Statement::If(
                        Box::new(ra::Expr::call_rt(
                            ra::RtItem::ScalarIntoBranchCondition,
                            [self.translate_expr(break_if, expr_ctx)?],
                        )),
                        ra::Block(vec![ra::Statement::Break(Some("naga_break"))], None),
                        ra::Block(vec![], None),
                    ));
                }

                vec![ra::Statement::Loop(
                    "naga_break",
                    ra::Block(rust_loop_body, None),
                )]
            }
            Statement::Break => vec![ra::Statement::Break(Some("naga_break"))],
            Statement::Continue => vec![ra::Statement::Break(Some("naga_continue"))],

            Statement::ControlBarrier(_) | Statement::MemoryBarrier(_) => {
                vec![self.unimplemented_stmt("barriers")?]
            }
            Statement::RayQuery { .. } | Statement::RayPipelineFunction(_) => {
                vec![self.unimplemented_stmt("raytracing")?]
            }
            Statement::SubgroupBallot { .. }
            | Statement::SubgroupCollectiveOperation { .. }
            | Statement::SubgroupGather { .. } => {
                vec![self.unimplemented_stmt("workgroup operations")?]
            }
            Statement::CooperativeStore { .. } => {
                vec![self.unimplemented_stmt("cooperative store")?]
            }
        })
    }

    /// Translate a statement which assigns `value_expr` to `*pointer`.
    ///
    /// This is a helper for [`Self::translate_statement()`], broken out because not all pointers
    /// will correspond to single Rust places of the correct type; sometimes we need to use setter
    /// functions, so this becomes potentially very complex.
    fn translate_store_statement(
        &mut self,
        expr_ctx: &ExpressionCtx<'_>,
        pointer: Handle<Expression>,
        value_expr: Handle<Expression>,
    ) -> Result<ra::Statement, Error> {
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
            return self.unimplemented_stmt("atomic operations");
        }

        if let Expression::AccessIndex { base, index } = *pointer_expr {
            let access_base_type = expr_ctx.resolve_type(base);
            let access_pointer_base_type = access_base_type
                .pointer_base_type()
                .expect("Store statement’s access expression's base type not a pointer type");

            // Decide whether to use an accessor function instead of an assignment.
            if let TypeInner::Vector { .. } = access_pointer_base_type.inner_with(expr_ctx.types())
            {
                return Ok(ra::Statement::Expr(ra::Expr::Method(
                    Box::new(self.expr_ast_with_indirection(base, expr_ctx, Indirection::Place)?),
                    Cow::Owned(format!("set_{}", back::COMPONENTS[index as usize])),
                    vec![self.translate_expr(value_expr, expr_ctx)?],
                )));
            }
        }

        // Fallthrough: Use Rust assignment.

        // The fields of aggregates are (currently) translated as `TypeTranslation::RustScalar`.
        // Therefore, if we are storing to a member of a struct, we need to insert a conversion.
        // TODO: this should be factored out into a general function for converting
        // between TypeTranslations.
        let value_expr = match TypeTranslation::from(pointer_type.pointer_space().unwrap()) {
            TypeTranslation::RustScalar => ra::Expr::call_rt(
                ra::RtItem::ScalarIntoInner,
                [self.translate_expr(value_expr, expr_ctx)?],
            ),
            TypeTranslation::ShaderScalar | TypeTranslation::Simd => {
                // No unwrapping
                self.translate_expr(value_expr, expr_ctx)?
            }
        };

        Ok(ra::Statement::Assign(
            self.expr_ast_with_indirection(pointer, expr_ctx, Indirection::Place)?,
            value_expr,
        ))
    }

    /// Build a `let` statement. Helper for statement processing.
    fn let_ast(
        &self,
        module: &Module,
        handle: Handle<Expression>,
        func_ctx: &back::FunctionCtx<'_>,
        name: String,
        expr: ra::Expr,
    ) -> Result<ra::Statement, Error> {
        let rust_ty = if self.config.flags.contains(WriterFlags::EXPLICIT_TYPES) {
            Some(match func_ctx.info[handle].ty {
                proc::TypeResolution::Handle(ty_handle) => {
                    self.type_ast(module, ty_handle, TypeTranslation::Simd)?
                }
                proc::TypeResolution::Value(ref inner) => {
                    self.type_ast_inner(module, inner, TypeTranslation::Simd)?
                }
            })
        } else {
            None
        };

        Ok(ra::Statement::Let(
            ra::Pattern::Binding(name),
            rust_ty,
            Some(expr),
        ))
    }

    /// Translate `expr` to Rust.
    ///
    /// See `expr_ast_with_indirection` for details.
    fn translate_expr(
        &self,
        expr: Handle<Expression>,
        expr_ctx: &ExpressionCtx<'_>,
    ) -> Result<ra::Expr, Error> {
        self.expr_ast_with_indirection(expr, expr_ctx, Indirection::Ordinary)
    }

    /// Translate multiple expressions.
    fn translate_exprs(
        &self,
        expr_ctx: &ExpressionCtx<'_>,
        exprs: impl IntoIterator<Item = Handle<Expression>>,
    ) -> Result<Vec<ra::Expr>, Error> {
        exprs
            .into_iter()
            .map(|expr| self.translate_expr(expr, expr_ctx))
            .collect()
    }

    /// Translate `expr` to Rust with the requested indirection.
    ///
    /// The `requested` argument indicates how the produced Rust expression’s type should relate
    /// to the Naga type of the input expression. See [`Indirection`]’s documentation for details.
    fn expr_ast_with_indirection(
        &self,
        expr: Handle<Expression>,
        expr_ctx: &ExpressionCtx<'_>,
        requested: Indirection,
    ) -> Result<ra::Expr, Error> {
        // If the plain form of the expression is not what we need, emit the
        // operator necessary to correct that.
        let (plain_expr, plain): (ra::Expr, Indirection) =
            self.expr_ast_plain_form(expr, expr_ctx)?;
        Ok(match (requested, plain) {
            // The plain form expression will be a place.
            // Convert it to a reference to match the Naga pointer type.
            //
            // Or, the plain form expression will be a Rust value that we want to take by
            // reference (currently, a texture handle; in the future, buffers too).
            //
            // TODO: We need to choose which borrow operator to use.
            (Indirection::Ordinary, Indirection::Place)
            | (Indirection::Ref, Indirection::Ordinary) => {
                ra::Expr::Borrow(ra::PtrKind::Shared(None), Box::new(plain_expr))
            }

            // The plain form expression will be a pointer, but the caller wants its pointee.
            // Insert a dereference operator.
            (Indirection::Place, Indirection::Ordinary)
            | (Indirection::Ordinary, Indirection::Ref) => ra::Expr::Deref(Box::new(plain_expr)),

            // Matches.
            (Indirection::Place, Indirection::Place)
            | (Indirection::Ordinary, Indirection::Ordinary)
            | (Indirection::Ref, Indirection::Ref) => plain_expr,

            (Indirection::Ref, Indirection::Place) | (Indirection::Place, Indirection::Ref) => {
                unreachable!("multi-level ref/deref")
            }
        })
    }

    /// Translates to the 'plain form' of `expr`.
    ///
    /// An expression's 'plain form' is the shortest rendition of that
    /// expression's meaning into Rust, lacking `&` or `*` operators.
    /// Therefore, it may not have a type which matches the Naga IR expression
    /// type (because Naga does not have places, only pointers and non-pointer values).
    ///
    /// When it does not match, this is indicated by the second part of the return value.
    /// It is the caller’s responsibility to adapt as needed, usually via
    /// [`Self::expr_ast_with_indirection()`].
    ///
    /// The return type of the written expression always follows [`TypeTranslation::Simd`] form.
    /// (We will need to refine that later.)
    fn expr_ast_plain_form(
        &self,
        expr: Handle<Expression>,
        expr_ctx: &ExpressionCtx<'_>,
    ) -> Result<(ra::Expr, Indirection), Error> {
        if let Some(name) = self.named_expressions.get(&expr) {
            return Ok((ra::Expr::Ident(name.clone()), Indirection::Ordinary));
        }

        let expression = &expr_ctx.expressions()[expr];
        let module = expr_ctx.module();

        // How the indirection of the expression we produce relates to the indirection expected
        // in the Naga IR. In particular, the below `match` will reset this to `Indirection::Place`
        // whenever we produce a Rust expression that is a Rust place, in a case where the Naga IR
        // is expecting the expression to yield a pointer value.
        let mut indirection = Indirection::Ordinary;

        let rust_expr = match *expression {
            Expression::Literal(literal) => {
                let rust_literal = match literal {
                    // TODO: Should we use the `half` library for f16 support at run time
                    // instead of only allowing it as a Rust unstable feature?
                    naga::Literal::F16(value) => ra::Expr::LitF16(value),
                    naga::Literal::F32(value) => ra::Expr::LitF32(value),
                    naga::Literal::U16(value) => ra::Expr::LitU16(value),
                    naga::Literal::U32(value) => ra::Expr::LitU32(value),
                    naga::Literal::I16(value) => ra::Expr::LitI16(value),
                    naga::Literal::I32(value) => ra::Expr::LitI32(value),
                    naga::Literal::Bool(value) => ra::Expr::LitBool(value),
                    naga::Literal::F64(value) => ra::Expr::LitF64(value),
                    naga::Literal::I64(value) => ra::Expr::LitI64(value),
                    naga::Literal::U64(value) => ra::Expr::LitU64(value),
                    naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_) => {
                        unreachable!(
                            "abstract types should not appear in IR presented to backends"
                        );
                    }
                };
                ra::Expr::call_rt(ra::RtItem::Scalar, [rust_literal])
            }
            Expression::Constant(handle) => {
                let constant = &module.constants[handle];
                if constant.name.is_some() {
                    ra::Expr::Ident(self.names[&NameKey::Constant(handle)].clone())
                } else {
                    self.translate_expr(constant.init, expr_ctx)?
                }
            }
            Expression::ZeroValue(_ty) => {
                // TODO: need to translate type
                ra::Expr::call_rt(ra::RtItem::ZeroFn, [])
            }
            Expression::Compose { ty, ref components } => {
                self.constructor_expression_ast(ty, components, expr_ctx)?
            }
            Expression::Splat { size, value } => ra::Expr::call_rt(
                ra::RtItem::SplatFromScalar(size),
                [self.translate_expr(value, expr_ctx)?],
            ),
            Expression::Override(_) => unreachable!(),
            Expression::FunctionArgument(pos) => {
                let name_key = expr_ctx.expect_func_ctx().argument_key(pos);
                let name = &self.names[&name_key];
                ra::Expr::Ident(name.clone())
            }
            Expression::Binary { op, left, right } => match BinOpClassified::from(op) {
                BinOpClassified::Vectorizable(_) => ra::Expr::BinOp(
                    Box::new(self.translate_expr(left, expr_ctx)?),
                    op,
                    Box::new(self.translate_expr(right, expr_ctx)?),
                ),
                BinOpClassified::ScalarBool(bop) => ra::Expr::Method(
                    Box::new(self.translate_expr(left, expr_ctx)?),
                    Cow::Borrowed(bop.to_vector_method()),
                    vec![self.translate_expr(right, expr_ctx)?],
                ),
                BinOpClassified::ShortCircuit(_bop) => {
                    // The ".0"s are for unwrapping the input `Scalar`s
                    // TODO: when we support SIMD this will need to change completely
                    ra::Expr::call_rt(
                        ra::RtItem::Scalar,
                        vec![ra::Expr::BinOp(
                            Box::new(ra::Expr::TupleField(
                                Box::new(self.translate_expr(left, expr_ctx)?),
                                0,
                            )),
                            op,
                            Box::new(ra::Expr::TupleField(
                                Box::new(self.translate_expr(right, expr_ctx)?),
                                0,
                            )),
                        )],
                    )
                }
            },
            Expression::Access { base, index } => {
                // In the Naga IR, `Access` and `AccessIndex` pass through the pointer-ness of
                // their `base` value, but in Rust, the result of `container[index]` is always
                // a place, so we need to report `Indirection::Place` to counteract that.
                let base_ty = expr_ctx.resolve_type(base);
                if let TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } = base_ty {
                    indirection = Indirection::Place;
                }

                // TODO: when we support SIMD, this will need to change to not be a single indexing
                // expression but a scatter/gather operation which isn’t a Rust place.
                ra::Expr::Index(
                    Box::new(self.expr_ast_with_indirection(base, expr_ctx, indirection)?),
                    Box::new(ra::Expr::call_rt(
                        ra::RtItem::ScalarIntoArrayIndex,
                        [self.translate_expr(index, expr_ctx)?],
                    )),
                )
            }
            Expression::AccessIndex { base, index } => {
                let result_ty = expr_ctx.resolve_type(expr);

                let base_ty_res = &expr_ctx.expect_func_ctx().info[base].ty;
                let mut base_ty_resolved = base_ty_res.inner_with(&module.types);

                // In the Naga IR, `Access` and `AccessIndex` pass through the pointer-ness of
                // their `base` value, but in Rust, the result of `container[index]` is always
                // a place, so we need to report `Indirection::Place` to counteract that.
                if let TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } = base_ty_resolved
                {
                    indirection = Indirection::Place;
                }

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
                    TypeInner::Vector { .. } => ra::Expr::Method(
                        Box::new(self.expr_ast_with_indirection(base, expr_ctx, indirection)?),
                        Cow::Borrowed(["x", "y", "z", "w"][index as usize]),
                        vec![],
                    ),
                    TypeInner::Matrix { .. }
                    | TypeInner::Array { .. }
                    | TypeInner::BindingArray { .. }
                    | TypeInner::ValuePointer { .. } => ra::Expr::Index(
                        Box::new(self.expr_ast_with_indirection(base, expr_ctx, indirection)?),
                        Box::new(ra::Expr::LitUsize(index)),
                    ),

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
                        // This will never panic in case the type is a `Struct`; this is not so
                        // for other types, so we can only check while inside this match arm
                        let ty = base_container_ty_handle.unwrap();
                        if element_type_is_scalar {
                            ra::Expr::call_rt(
                                ra::RtItem::Scalar,
                                [ra::Expr::NamedField(
                                    Box::new(self.expr_ast_with_indirection(
                                        base,
                                        expr_ctx,
                                        indirection,
                                    )?),
                                    self.names[&NameKey::StructMember(ty, index)].clone(),
                                )],
                            )
                        } else {
                            ra::Expr::NamedField(
                                Box::new(self.expr_ast_with_indirection(
                                    base,
                                    expr_ctx,
                                    indirection,
                                )?),
                                self.names[&NameKey::StructMember(ty, index)].clone(),
                            )
                        }
                    }
                    ref other => unreachable!("cannot index into a {other:?}"),
                }
            }
            Expression::ImageSample { .. } => {
                self.unimplemented_expr("texture sampling (other than textureLoad)")?
            }
            Expression::ImageQuery { image, query } => {
                let image_expr =
                    self.expr_ast_with_indirection(image, expr_ctx, Indirection::Ref)?;
                match query {
                    naga::ImageQuery::Size { level } => ra::Expr::call_rt(
                        ra::RtItem::TextureDimensions,
                        [
                            image_expr,
                            if let Some(level) = level {
                                ra::Expr::call_rt(
                                    ra::RtItem::ScalarIntoInner,
                                    [self.translate_expr(level, expr_ctx)?],
                                )
                            } else {
                                ra::Expr::LitI32(0)
                            },
                        ],
                    ),
                    naga::ImageQuery::NumLevels => ra::Expr::call_rt(
                        ra::RtItem::TextureNzToScalar,
                        [ra::Expr::call_rt(
                            ra::RtItem::TextureNumLevels,
                            [image_expr],
                        )],
                    ),
                    naga::ImageQuery::NumLayers => ra::Expr::call_rt(
                        ra::RtItem::TextureNzToScalar,
                        [ra::Expr::call_rt(
                            ra::RtItem::TextureNumLayers,
                            [image_expr],
                        )],
                    ),
                    naga::ImageQuery::NumSamples => ra::Expr::call_rt(
                        ra::RtItem::TextureNzToScalar,
                        [ra::Expr::call_rt(
                            ra::RtItem::TextureNumSamples,
                            [image_expr],
                        )],
                    ),
                }
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => ra::Expr::call_rt(
                ra::RtItem::TextureLoad,
                [
                    self.expr_ast_with_indirection(image, expr_ctx, Indirection::Ref)?,
                    self.translate_expr(coordinate, expr_ctx)?,
                    if let Some(array_index) = array_index {
                        ra::Expr::call_rt(
                            ra::RtItem::ScalarIntoInner,
                            [self.translate_expr(array_index, expr_ctx)?],
                        )
                    } else {
                        ra::Expr::LitI32(0)
                    },
                    if let Some(sample) = sample {
                        ra::Expr::call_rt(
                            ra::RtItem::ScalarIntoInner,
                            [self.translate_expr(sample, expr_ctx)?],
                        )
                    } else {
                        ra::Expr::LitI32(0)
                    },
                    if let Some(level) = level {
                        ra::Expr::call_rt(
                            ra::RtItem::ScalarIntoInner,
                            [self.translate_expr(level, expr_ctx)?],
                        )
                    } else {
                        ra::Expr::LitI32(0)
                    },
                ],
            ),
            Expression::GlobalVariable(handle) => {
                let global = &module.global_variables[handle];

                // The plain form of `GlobalVariable(g)` is `self.g`, which is a
                // Rust place. However, globals in the `Handle` address space are immutable,
                // and `GlobalVariable` expressions for those produce the value directly,
                // not a pointer to it. Therefore, such expressions have `Indirection::Place`.
                // (Note that the exception for Handle is a fact about Naga IR, not this backend.)
                indirection = match global.space {
                    naga::AddressSpace::Handle => Indirection::Ordinary,
                    _ => Indirection::Place,
                };

                ra::Expr::NamedField(
                    Box::new(self.config.global_field_access_expr(global)),
                    self.names[&NameKey::GlobalVariable(handle)].clone(),
                )
            }

            Expression::As {
                expr,
                kind: to_kind,
                convert: to_width,
            } => {
                use naga::ScalarKind as Sk;
                use naga::TypeInner as Ti;

                let input_type = expr_ctx.resolve_type(expr);
                let rust_expr = self.translate_expr(expr, expr_ctx)?;

                match (input_type, to_kind, to_width) {
                    (
                        Ti::Vector { size: _, scalar: _ } | Ti::Scalar(_),
                        to_kind,
                        Some(to_width),
                    ) => ra::Expr::Method(
                        Box::new(rust_expr),
                        Cow::Borrowed(match (to_kind, to_width) {
                            (Sk::Sint, 4) => "cast_elem_as_i32",
                            (Sk::Sint, 8) => "cast_elem_as_i64",
                            (Sk::Uint, 4) => "cast_elem_as_u32",
                            (Sk::Uint, 8) => "cast_elem_as_u64",
                            (Sk::Float, 4) => "cast_elem_as_f32",
                            (Sk::Float, 8) => "cast_elem_as_f64",
                            _ => panic!(
                                "unimplemented cast of vector to kind {to_kind:?} width {to_width:?}"
                            ),
                        }),
                        vec![],
                    ),
                    _ => panic!(
                        "unimplemented cast {input_type:?} to kind {to_kind:?} width {to_width:?}"
                    ),
                }
            }
            Expression::Load { pointer } => {
                self.expr_ast_with_indirection(pointer, expr_ctx, Indirection::Place)?
            }
            Expression::LocalVariable(handle) => {
                // In Naga, a `LocalVariable(x)` expression produces a pointer,
                // but our plain form is a variable name `x`,
                // which means the caller must reference it if desired.
                indirection = Indirection::Place;

                ra::Expr::Ident(self.names[&expr_ctx.expect_func_ctx().name_key(handle)].clone())
            }
            Expression::ArrayLength(expr) => ra::Expr::Method(
                Box::new(self.translate_expr(expr, expr_ctx)?),
                Cow::Borrowed("len"),
                vec![],
            ),

            Expression::Math {
                fun,
                arg: first_arg,
                arg1,
                arg2,
                arg3,
            } => ra::Expr::Method(
                Box::new(self.translate_expr(first_arg, expr_ctx)?),
                Cow::Borrowed(conv::math_function_to_method(fun)),
                self.translate_exprs(
                    expr_ctx,
                    [arg1, arg2, arg3].into_iter().flatten(), // flatten options into nonexistence
                )?,
            ),

            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let swizzle_method = String::from_iter(
                    pattern[..size as usize]
                        .iter()
                        .map(|&component| back::COMPONENTS[component as usize]),
                );
                ra::Expr::Method(
                    Box::new(self.translate_expr(vector, expr_ctx)?),
                    Cow::Owned(swizzle_method),
                    vec![],
                )
            }
            Expression::Unary { op, expr } => {
                let ctor = match op {
                    naga::UnaryOperator::Negate => ra::Expr::Negate,
                    naga::UnaryOperator::LogicalNot => ra::Expr::Not,
                    naga::UnaryOperator::BitwiseNot => ra::Expr::Not,
                };
                ctor(Box::new(self.translate_expr(expr, expr_ctx)?))
            }

            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                // Calls {vector type}::select() method
                ra::Expr::Method(
                    Box::new(self.translate_expr(reject, expr_ctx)?),
                    Cow::Borrowed("select"),
                    vec![
                        self.translate_expr(accept, expr_ctx)?,
                        self.translate_expr(condition, expr_ctx)?,
                    ],
                )
            }
            Expression::Derivative { .. } => self.unimplemented_expr("derivatives")?,
            Expression::Relational { fun, argument } => {
                use naga::RelationalFunction as Rf;

                match fun {
                    Rf::All => ra::Expr::Method(
                        Box::new(self.translate_expr(argument, expr_ctx)?),
                        Cow::Borrowed("all"),
                        vec![],
                    ),
                    Rf::Any => ra::Expr::Method(
                        Box::new(self.translate_expr(argument, expr_ctx)?),
                        Cow::Borrowed("any"),
                        vec![],
                    ),
                    Rf::IsNan => self.unimplemented_expr("IsNan")?,
                    Rf::IsInf => self.unimplemented_expr("IsInf")?,
                }
            }
            Expression::RayQueryGetIntersection { .. }
            | Expression::RayQueryVertexPositions { .. }
            | Expression::CooperativeLoad { .. }
            | Expression::CooperativeMultiplyAdd { .. } => unreachable!("unsupported feature"),
            //
            Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::RayQueryProceedResult
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. } => {
                unreachable!("nothing to do here, since call expression already cached")
            }
        };

        Ok((rust_expr, indirection))
    }

    /// Translates [`Expression::Compose`].
    /// Examines the type to write an appropriate constructor or literal expression for it.
    ///
    /// We do not delegate to a library trait for this because the construction
    /// must be const-compatible.
    fn constructor_expression_ast(
        &self,
        ty: Handle<naga::Type>,
        components: &[Handle<Expression>],
        expr_ctx: &ExpressionCtx<'_>,
    ) -> Result<ra::Expr, Error> {
        use naga::VectorSize::{Bi, Quad, Tri};

        let translated_components = components
            .iter()
            .map(|&component| self.translate_expr(component, expr_ctx))
            .collect::<Result<Vec<ra::Expr>, Error>>()?;

        let ctor_name: &'static str = match expr_ctx.types()[ty].inner {
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
                return Ok(ra::Expr::Array(translated_components));
            }

            // Fallback: Assume that a suitable `T::new()` associated function
            // exists.
            _ => "new",
        };

        Ok(ra::Expr::Call(
            Box::new(ra::Expr::QualifiedPath(
                self.type_ast(expr_ctx.module(), ty, TypeTranslation::Simd)?,
                ctor_name,
            )),
            translated_components,
        ))
    }

    /// Translate the gtiven `type_handle` to [`ra::Type`] format.
    ///
    /// The form a type takes depends on the address space in which the value of that type lives.
    pub(super) fn type_ast(
        &self,
        module: &Module,
        type_handle: Handle<naga::Type>,
        type_translation: TypeTranslation,
    ) -> Result<ra::Type, Error> {
        match module.types[type_handle].inner {
            TypeInner::Struct { .. } => Ok(ra::Type::User(
                self.names[&NameKey::Type(type_handle)].clone(),
                ra::Generics::None,
            )),
            ref other => self.type_ast_inner(module, other, type_translation),
        }
    }

    fn type_ast_inner(
        &self,
        module: &Module,
        inner: &TypeInner,
        type_translation: TypeTranslation,
    ) -> Result<ra::Type, Error> {
        Ok(match *inner {
            TypeInner::Vector { size, scalar } => {
                ra::Type::RtGen(ra::RtGen::vector(size), scalar.try_into()?)
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => ra::Type::RtGen(ra::RtGen::matrix(columns, rows), scalar.try_into()?),
            TypeInner::Scalar(scalar) => match type_translation {
                TypeTranslation::RustScalar => ra::Type::BareScalar(scalar.try_into()?),
                TypeTranslation::ShaderScalar | TypeTranslation::Simd => {
                    ra::Type::RtGen(ra::RtGen::Scalar, scalar.try_into()?)
                }
            },

            TypeInner::Sampler { comparison: false } => ra::Type::Sampler,
            TypeInner::Sampler { comparison: true } => ra::Type::SamplerComparison,
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                let (scalar_type, multisampled): (ra::Scalar, bool) = match class {
                    naga::ImageClass::Sampled { kind, multi } => (
                        match kind {
                            naga::ScalarKind::Sint => ra::Scalar::I32,
                            naga::ScalarKind::Uint => ra::Scalar::U32,
                            naga::ScalarKind::Float => ra::Scalar::F32,
                            naga::ScalarKind::Bool => ra::Scalar::Bool,
                            naga::ScalarKind::AbstractInt | naga::ScalarKind::AbstractFloat => {
                                unreachable!(
                                    "abstract types should not appear in IR presented to backends"
                                )
                            }
                        },
                        multi,
                    ),
                    naga::ImageClass::Depth { multi } => (ra::Scalar::F32, multi),
                    naga::ImageClass::External => (ra::Scalar::F32, false),
                    naga::ImageClass::Storage { .. } => {
                        return Err(Error::Unimplemented("storage texture types".into()));
                    }
                };
                ra::Type::Texture {
                    dim,
                    multisampled,
                    arrayed,
                    storage_type: Box::new(ra::Type::Ptr(
                        ra::PtrKind::Shared(Some("g")),
                        // TODO: we want to support fully statically dispatched texture access,
                        // but that will require more work to either:
                        //
                        // * allow the user to specify a concrete type for the texture storage,
                        // * make the resource struct generic,
                        // * or allow the user to provide their own resource struct (possibly
                        //   corresponding to a single bind group) and adapt to its types.
                        //
                        // `dyn` is a placeholder for further work, and it’s not great to go through
                        // a dynamic dispatch for every texel load.
                        Box::new(ra::Type::DynTextureRead {
                            dim,
                            scalar: scalar_type,
                        }),
                    )),
                }
            }
            TypeInner::Atomic(scalar) => ra::Type::Atomic(scalar.try_into()?),
            TypeInner::Array {
                base,
                size,
                stride: _,
            } => {
                let element_type = Box::new(self.type_ast(module, base, type_translation)?);
                match size {
                    naga::ArraySize::Constant(size) => ra::Type::Array(element_type, size.get()),
                    naga::ArraySize::Pending(_handle) => {
                        return Err(Error::Unimplemented("override array size".into()));
                    }
                    naga::ArraySize::Dynamic => ra::Type::Slice(element_type),
                }
            }
            TypeInner::BindingArray { .. } => {
                return Err(Error::Unimplemented("binding array".into()));
            }
            TypeInner::Pointer {
                base,
                space: pointee_space,
            } => ra::Type::Ptr(
                if self.config.flags.contains(WriterFlags::RAW_POINTERS) {
                    ra::PtrKind::RawMut
                } else {
                    ra::PtrKind::Exclusive(None)
                },
                Box::new(self.type_ast(module, base, TypeTranslation::from(pointee_space))?),
            ),
            TypeInner::ValuePointer {
                size: _,
                scalar: _,
                space: _,
            } => todo!(),
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
        })
    }

    fn translate_global_variable_to_struct_field(
        &self,
        module: &Module,
        global: &naga::GlobalVariable,
        handle: Handle<naga::GlobalVariable>,
    ) -> Result<ra::Field, Error> {
        let &naga::GlobalVariable {
            name: _, // renamed instead
            space,
            binding: _, // don't (yet) expose numeric binding locations
            ty,
            init: _,
            memory_decorations: _, // TODO: probably need to do things with this
        } = global;

        // TODO: reenable this
        // // Note bindings.
        // // These are not emitted as attributes because Rust does not allow macro attributes to be
        // // placed on struct fields.
        // if let Some(naga::ResourceBinding { group, binding }) = global.binding {
        //     writeln!(out, "{INDENT}// group({group}) binding({binding})")?;
        // }

        Ok(ra::Field {
            attributes: if let Some(naga::ResourceBinding { group, binding }) = global.binding {
                vec![ra::Attribute::Doc(format!(
                    "group({group}) binding({binding})"
                ))]
            } else {
                vec![]
            },
            visibility: self.visibility(),
            name: self.names[&NameKey::GlobalVariable(handle)].clone(),
            ty: self.type_ast(module, ty, TypeTranslation::from(space))?,
        })
    }
    fn global_variable_as_field_initializer_expr(
        &self,
        module: &Module,
        info: &ModuleInfo,
        global: &naga::GlobalVariable,
    ) -> Result<ra::Expr, Error> {
        if let Some(init) = global.init {
            self.translate_expr(
                init,
                &ExpressionCtx::Global {
                    expressions: &module.global_expressions,
                    module,
                    module_info: info,
                },
            )
        } else {
            Ok(ra::Expr::call_rt(ra::RtItem::ZeroFn, []))
        }
    }

    /// Translates a [`naga::Constant`] to a Rust `const` item.
    fn translate_global_constant(
        &self,
        module: &Module,
        info: &ModuleInfo,
        handle: Handle<naga::Constant>,
    ) -> Result<ra::ConstItem, Error> {
        let name = self.names[&NameKey::Constant(handle)].clone();
        let visibility = self.visibility();
        let ty = self.type_ast(
            module,
            module.constants[handle].ty,
            TypeTranslation::ShaderScalar,
        )?;
        let init = module.constants[handle].init;

        Ok(ra::ConstItem {
            attributes: vec![ra::Attribute::AllowNonUpperCaseGlobals],
            visibility,
            name,
            ty,
            value: self.translate_expr(
                init,
                &ExpressionCtx::Global {
                    expressions: &module.global_expressions,
                    module,
                    module_info: info,
                },
            )?,
        })
    }

    /// For a feature that naga-rust does not support, either return an immediate conversion error, or emit
    /// an expression that panics when executed.
    fn unimplemented_expr(&self, unimplemented_feature: &'static str) -> Result<ra::Expr, Error> {
        assert!(
            !unimplemented_feature.contains(['{', '}']),
            "escaping format strings not supported yet"
        );

        if self.config.flags.contains(WriterFlags::ALLOW_UNIMPLEMENTED) {
            Ok(ra::Expr::FormatLikeMacro(
                // TODO: path needs to be qualified
                "unimplemented",
                alloc::format!(
                    "this shader function contains a feature which \
                    cannot yet be translated to Rust, {unimplemented_feature}"
                ),
            ))
        } else {
            Err(Error::Unimplemented(unimplemented_feature.into()))
        }
    }

    fn unimplemented_stmt(
        &self,
        unimplemented_feature: &'static str,
    ) -> Result<ra::Statement, Error> {
        Ok(ra::Statement::Expr(
            self.unimplemented_expr(unimplemented_feature)?,
        ))
    }

    fn visibility(&self) -> ra::Visibility {
        if self.config.flags.contains(WriterFlags::PUBLIC) {
            ra::Visibility::Public
        } else {
            ra::Visibility::Private
        }
    }
}
