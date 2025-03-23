use alloc::{
    string::{String, ToString},
    vec,
};
use arrayvec::ArrayVec;
use core::fmt::Write;

use naga::{
    Expression, Handle, Module, Scalar, ShaderStage, TypeInner,
    back::{self, INDENT},
    proc::{self, ExpressionKindTracker, NameKey},
    valid::ModuleInfo,
};

use crate::config::WriterFlags;
use crate::conv::{self, BinOpClassified, KEYWORDS_2024, SHADER_LIB, unwrap_to_rust};
use crate::util::{Gensym, LevelNext};
use crate::{Config, Error};

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

/// Edition of Rust code to generate.
///
/// We currently only support one edition, but this exists anyway to prepare to document
/// any edition dependencies in the code generator.
#[derive(Clone, Copy, Debug)]
enum Edition {
    Rust2024,
}

/// [`naga`] backend allowing you to translate shader code in any language supported by Naga
/// to Rust code.
///
/// A `Writer` stores a [`Config`] and data structures that can be reused for writing multiple
/// modules.
#[allow(missing_debug_implementations, reason = "TODO")]
pub struct Writer {
    config: Config,
    #[allow(dead_code)]
    edition: Edition,
    names: naga::FastHashMap<NameKey, String>,
    namer: proc::Namer,
    named_expressions: naga::FastIndexMap<Handle<Expression>, String>,
}

impl Writer {
    /// Creates a new [`Writer`].
    #[must_use]
    pub fn new(config: Config) -> Self {
        Writer {
            config,
            edition: Edition::Rust2024,
            names: naga::FastHashMap::default(),
            namer: proc::Namer::default(),
            named_expressions: naga::FastIndexMap::default(),
        }
    }

    fn reset(&mut self, module: &Module) {
        let Self {
            config,
            edition: _,
            names,
            namer,
            named_expressions,
        } = self;
        names.clear();
        namer.reset(
            module,
            KEYWORDS_2024,
            &[SHADER_LIB],
            &[],
            &[],
            &mut self.names,
        );
        if let Some(g) = &config.global_struct {
            // TODO: We actually want to say “treat this as reserved but do not rename it”,
            // but Namer doesn’t have that option
            namer.call(g);
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
            return Err(Error::Unimplemented("pipeline constants".into()));
        }

        self.reset(module);

        // Write top-level attributes
        write!(
            out,
            "\
                #[allow(dead_code, clippy::unnecessary_self_imports)]\n\
                use {runtime_path}::{{self as rt}};\n\
            ",
            runtime_path = self.config.runtime_path,
        )?;

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

        // If we are using global variables, write the `struct` that contains them.
        if let Some(global_struct) = self.config.global_struct.clone() {
            writeln!(out, "struct {global_struct} {{")?;
            for (handle, global) in module.global_variables.iter() {
                self.write_global_variable_as_struct_field(out, module, global, handle)?;
            }
            // TODO: instead of trying to implement Default, make a constructor function
            // for all globals that use bindings rather than initializers
            writeln!(
                out,
                "}}\n\
                impl Default for {global_struct} {{\n\
                {INDENT}fn default() -> Self {{ Self {{"
            )?;
            for (handle, global) in module.global_variables.iter() {
                self.write_global_variable_as_field_initializer(out, module, info, global, handle)?;
            }
            writeln!(out, "{INDENT}}}}}\n}}")?;

            // Start the `impl` block of the functions
            writeln!(out, "impl {global_struct} {{")?;
        } else if let Some((_, example)) = module.global_variables.iter().next() {
            return Err(Error::GlobalVariablesNotEnabled {
                example: example.name.clone().unwrap_or_default(),
            });
        }

        // Write all regular functions (which may or may not be in the `impl` block from above).
        for (handle, function) in module.functions.iter() {
            let fun_info = &info[handle];

            let func_ctx = back::FunctionCtx {
                ty: back::FunctionType::Function(handle),
                info: fun_info,
                expressions: &function.expressions,
                named_expressions: &function.named_expressions,
                expr_kind_tracker: ExpressionKindTracker::from_arena(&function.expressions),
            };

            // Write the function
            self.write_function(out, module, info, function, &func_ctx)?;

            writeln!(out)?;
        }

        // Write all entry points
        for (index, ep) in module.entry_points.iter().enumerate() {
            let attributes = match ep.stage {
                ShaderStage::Vertex | ShaderStage::Fragment => vec![Attribute::Stage(ep.stage)],
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
                expr_kind_tracker: ExpressionKindTracker::from_arena(&ep.function.expressions),
            };
            self.write_function(out, module, info, &ep.function, &func_ctx)?;

            if index < module.entry_points.len() - 1 {
                writeln!(out)?;
            }
        }

        if self.config.use_global_struct() {
            // End the `impl` block
            writeln!(out, "}}")?;
        }

        Ok(())
    }

    /// # Notes
    /// Ends in a newline
    fn write_function(
        &mut self,
        out: &mut dyn Write,
        module: &Module,
        info: &ModuleInfo,
        func: &naga::Function,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        self.write_attributes(out, back::Level(0), &[Attribute::AllowFunctionBody])?;

        // Write start of function item
        let func_name = match func_ctx.ty {
            back::FunctionType::EntryPoint(index) => &self.names[&NameKey::EntryPoint(index)],
            back::FunctionType::Function(handle) => &self.names[&NameKey::Function(handle)],
        };
        let visibility = self.visibility();
        write!(out, "{visibility}fn {func_name}(")?;

        if self.config.use_global_struct() {
            // TODO: need to figure out whether &mut is needed
            write!(out, "&self, ")?;
        } else if func_ctx.info.global_variable_count() > 0 {
            unreachable!(
                "function has globals but globals are not enabled; \
                should have been rejected earlier"
            );
        }

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
            self.write_type(out, module, arg.ty)?;
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
            self.write_type(out, module, result.ty)?;
        }

        write!(out, " {{")?;
        writeln!(out)?;

        // Write function local variables
        for (handle, local) in func.local_variables.iter() {
            // Write indentation (only for readability)
            write!(out, "{INDENT}")?;

            // Write the local name
            // The leading space is important
            write!(out, "let mut {}: ", self.names[&func_ctx.name_key(handle)])?;

            // Write the local type
            self.write_type(out, module, local.ty)?;

            // Write the local initializer if needed
            if let Some(init) = local.init {
                write!(out, " = ")?;
                self.write_expr(out, module, info, init, func_ctx)?;
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
            self.write_stmt(out, module, info, sta, func_ctx, back::Level(1))?;
        }

        writeln!(out, "}}")?;

        self.named_expressions.clear();

        Ok(())
    }

    /// Writes one or more [`Attribute`]s as outer attributes.
    fn write_attributes(
        &self,
        out: &mut dyn Write,
        level: back::Level,
        attributes: &[Attribute],
    ) -> BackendResult {
        for attribute in attributes {
            write!(out, "{level}#[")?;
            match *attribute {
                Attribute::AllowFunctionBody => {
                    write!(
                        out,
                        // `clippy::all` refers to all *default* clippy lints, not all clippy lints.
                        // We’re allowing all clippy categories except for restriction, which we
                        // shall assume is on purpose, and cargo, which is irrelvant.
                        "allow(unused_parens, clippy::all, clippy::pedantic, clippy::nursery)"
                    )?;
                }
                Attribute::Stage(shader_stage) => {
                    let stage_str = match shader_stage {
                        ShaderStage::Vertex => "vertex",
                        ShaderStage::Fragment => "fragment",
                        ShaderStage::Compute => "compute",
                    };
                    write!(out, "{SHADER_LIB}::{stage_str}")?;
                }
                Attribute::WorkGroupSize(size) => {
                    write!(
                        out,
                        "{SHADER_LIB}::workgroup_size({}, {}, {})",
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
        handle: Handle<naga::Type>,
        members: &[naga::StructMember],
    ) -> BackendResult {
        // TODO: we will need to do custom dummy fields to ensure that vec3s have correct alignment.
        let visibility = self.visibility();
        write!(
            out,
            "#[repr(C)]\n\
            {visibility}struct {}",
            self.names[&NameKey::Type(handle)]
        )?;
        write!(out, " {{")?;
        writeln!(out)?;
        for (index, member) in members.iter().enumerate() {
            // The indentation is only for readability
            write!(out, "{INDENT}")?;
            // if let Some(ref binding) = member.binding {
            //     self.write_attributes(&map_binding_to_attribute(binding))?;
            // }
            // Write struct member name and type
            let member_name =
                &self.names[&NameKey::StructMember(handle, index.try_into().unwrap())];
            write!(out, "{visibility}{member_name}: ")?;
            self.write_type(out, module, member.ty)?;
            write!(out, ",")?;
            writeln!(out)?;
        }

        writeln!(out, "}}")?;

        Ok(())
    }

    /// Helper method used to write statements
    ///
    /// # Notes
    /// Always adds a newline
    fn write_stmt(
        &mut self,
        out: &mut dyn Write,
        module: &Module,
        info: &ModuleInfo,
        stmt: &naga::Statement,
        func_ctx: &back::FunctionCtx<'_>,
        level: back::Level,
    ) -> BackendResult {
        use naga::{Expression, Statement};

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
                        self.write_expr(out, module, info, handle, func_ctx)?;
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

                write!(out, "{level}if ")?;
                self.write_expr(out, module, info, condition, func_ctx)?;
                writeln!(out, " {{")?;
                for s in accept {
                    self.write_stmt(out, module, info, s, func_ctx, l2)?;
                }
                if !reject.is_empty() {
                    writeln!(out, "{level}}} else {{")?;
                    for s in reject {
                        self.write_stmt(out, module, info, s, func_ctx, l2)?;
                    }
                }
                writeln!(out, "{level}}}")?
            }
            Statement::Return { value } => {
                write!(out, "{level}return")?;
                if let Some(return_value) = value {
                    write!(out, " ")?;
                    self.write_expr(out, module, info, return_value, func_ctx)?;
                }
                writeln!(out, ";")?;
            }
            Statement::Kill => write!(out, "{level}{SHADER_LIB}::discard();")?,
            Statement::Store { pointer, value } => {
                let is_atomic_pointer = func_ctx
                    .resolve_type(pointer, &module.types)
                    .is_atomic_pointer(&module.types);

                if is_atomic_pointer {
                    return Err(Error::Unimplemented("atomic operations".into()));
                }

                write!(out, "{level}")?;
                // We have a Naga “pointer” but it might actually denote a plain variable.
                // We ask for `Indirection::Place` to say: please dereference the logical pointer
                // and give me the dereference op *or* plain variable to assign to.
                self.write_expr_with_indirection(
                    out,
                    module,
                    info,
                    pointer,
                    func_ctx,
                    Indirection::Place,
                )?;
                write!(out, " = ")?;
                self.write_expr(out, module, info, value, func_ctx)?;

                writeln!(out, ";")?
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

                // If we are using a global struct, then functions are methods of that struct.
                if self.config.use_global_struct() {
                    write!(out, "self.")?;
                }

                let func_name = &self.names[&NameKey::Function(function)];
                write!(out, "{func_name}(")?;
                for (index, &argument) in arguments.iter().enumerate() {
                    if index != 0 {
                        write!(out, ", ")?;
                    }
                    self.write_expr(out, module, info, argument, func_ctx)?;
                }
                writeln!(out, ");")?
            }
            Statement::Atomic { .. } => {
                return Err(Error::Unimplemented("atomic operations".into()));
            }
            Statement::ImageAtomic { .. } => {
                return Err(Error::TexturesAreUnsupported {
                    found: "textureAtomic",
                });
            }
            Statement::WorkGroupUniformLoad { .. } => {
                todo!("Statement::WorkGroupUniformLoad");
            }
            Statement::ImageStore { .. } => {
                return Err(Error::TexturesAreUnsupported {
                    found: "textureStore",
                });
            }
            Statement::Block(ref block) => {
                write!(out, "{level}")?;
                writeln!(out, "{{")?;
                for s in block.iter() {
                    self.write_stmt(out, module, info, s, func_ctx, level.next())?;
                }
                writeln!(out, "{level}}}")?;
            }
            Statement::Switch {
                selector,
                ref cases,
            } => {
                // Beginning of the match expression
                write!(out, "{level}")?;
                write!(out, "match ")?;
                self.write_expr(out, module, info, selector, func_ctx)?;
                writeln!(out, " {{")?;

                // Generate each arm, collapsing empty fall-through into a single arm.
                let l2 = level.next();
                let mut new_match_arm = true;
                for case in cases {
                    if case.fall_through && !case.body.is_empty() {
                        // TODO
                        return Err(Error::Unimplemented(
                            "fall-through switch case block".into(),
                        ));
                    }

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

                    new_match_arm = !case.fall_through;

                    // End this pattern and begin the body of this arm,
                    // if it is not fall-through.
                    if new_match_arm {
                        writeln!(out, " => {{")?;
                        for sta in case.body.iter() {
                            self.write_stmt(out, module, info, sta, func_ctx, l2.next())?;
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
                write!(out, "{level}")?;
                writeln!(out, "loop {{")?;

                let l2 = level.next();
                for sta in body.iter() {
                    self.write_stmt(out, module, info, sta, func_ctx, l2)?;
                }

                if !continuing.is_empty() {
                    return Err(Error::Unimplemented("continuing".into()));
                }
                if break_if.is_some() {
                    return Err(Error::Unimplemented("break_if".into()));
                }

                writeln!(out, "{level}}}")?;
            }
            Statement::Break => writeln!(out, "{level}break;")?,
            Statement::Continue => writeln!(out, "{level}continue;")?,
            Statement::Barrier(_) => {
                return Err(Error::Unimplemented("barriers".into()));
            }
            Statement::RayQuery { .. } => {
                return Err(Error::Unimplemented("raytracing".into()));
            }
            Statement::SubgroupBallot { .. }
            | Statement::SubgroupCollectiveOperation { .. }
            | Statement::SubgroupGather { .. } => {
                return Err(Error::Unimplemented("workgroup operations".into()));
            }
        }

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
        module: &Module,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> Indirection {
        use naga::Expression as Ex;

        // Named expressions are `let` bindings.
        // so if their type is a Naga pointer, then that must be a Rust pointer
        // as well.
        if self.named_expressions.contains_key(&expr) {
            return Indirection::Ordinary;
        }

        match func_ctx.expressions[expr] {
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
                let global = &module.global_variables[handle];
                match global.space {
                    naga::AddressSpace::Handle => Indirection::Ordinary,
                    _ => Indirection::Place,
                }
            }

            // `Access` and `AccessIndex` pass through the pointer-ness of their `base` value.
            Ex::Access { base, .. } | Ex::AccessIndex { base, .. } => {
                let base_ty = func_ctx.resolve_type(base, &module.types);
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
                    self.write_type(out, module, ty_handle)?;
                }
                proc::TypeResolution::Value(ref inner) => {
                    self.write_type_inner(out, module, inner)?;
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
        module: &Module,
        info: &ModuleInfo,
        expr: Handle<Expression>,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        self.write_expr_with_indirection(out, module, info, expr, func_ctx, Indirection::Ordinary)
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
        module: &Module,
        info: &ModuleInfo,
        expr: Handle<Expression>,
        func_ctx: &back::FunctionCtx<'_>,
        requested: Indirection,
    ) -> BackendResult {
        // If the plain form of the expression is not what we need, emit the
        // operator necessary to correct that.
        let plain = self.plain_form_indirection(expr, module, func_ctx);
        match (requested, plain) {
            // The plain form expression will be a place.
            // Convert it to a reference to match the Naga pointer type.
            // TODO: We need to choose which borrow operator to use.
            (Indirection::Ordinary, Indirection::Place) => {
                write!(out, "(&")?;
                self.write_expr_plain_form(out, module, info, expr, func_ctx, plain)?;
                write!(out, ")")?;
            }

            // The plain form expression will be a pointer, but the caller wants its pointee.
            // Insert a dereference operator.
            (Indirection::Place, Indirection::Ordinary) => {
                write!(out, "(*")?;
                self.write_expr_plain_form(out, module, info, expr, func_ctx, plain)?;
                write!(out, ")")?;
            }
            // Matches.
            (Indirection::Place, Indirection::Place)
            | (Indirection::Ordinary, Indirection::Ordinary) => {
                self.write_expr_plain_form(out, module, info, expr, func_ctx, plain)?
            }
        }

        Ok(())
    }

    fn write_const_expression(
        &self,
        out: &mut dyn Write,
        module: &Module,
        info: &ModuleInfo,
        expr: Handle<Expression>,
    ) -> BackendResult {
        self.write_possibly_const_expression(
            out,
            module,
            info,
            expr,
            &module.global_expressions,
            |out, expr| self.write_const_expression(out, module, info, expr),
            |expr| info[expr].inner_with(&module.types),
        )
    }

    /// Writes an expression of the kinds that can appear both in constants and in function bodies.
    ///
    /// The difference between these two cases is that function bodies have access to a
    /// [`back::FunctionCtx`], but this doesn’t.
    //
    // Arguably we could do a runtime check for that instead of the `unreachable!()`s
    // on matching the expression enum.
    #[allow(clippy::too_many_arguments)]
    fn write_possibly_const_expression<'a>(
        &self,
        out: &mut dyn Write,
        module: &Module,
        info: &ModuleInfo,
        expr: Handle<Expression>,
        expressions: &naga::Arena<Expression>,
        write_expression: impl Copy + Fn(&mut dyn Write, Handle<Expression>) -> BackendResult,
        expression_type: impl Copy + Fn(Handle<Expression>) -> &'a TypeInner,
    ) -> BackendResult {
        match expressions[expr] {
            Expression::Literal(literal) => match literal {
                naga::Literal::F32(value) => write!(out, "{value}f32")?,
                naga::Literal::U32(value) => write!(out, "{value}u32")?,
                naga::Literal::I32(value) => {
                    write!(out, "{value}i32")?;
                }
                naga::Literal::Bool(value) => write!(out, "{value}")?,
                naga::Literal::F64(value) => write!(out, "{value}f64")?,
                naga::Literal::I64(value) => {
                    write!(out, "{value}i64")?;
                }
                naga::Literal::U64(value) => write!(out, "{value}u64")?,
                naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_) => {
                    unreachable!("abstract types should not appear in IR presented to backends");
                }
            },
            Expression::Constant(handle) => {
                let constant = &module.constants[handle];
                if constant.name.is_some() {
                    write!(out, "{}", self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.write_const_expression(out, module, info, constant.init)?;
                }
            }
            Expression::ZeroValue(ty) => {
                write!(out, "{SHADER_LIB}::zero::<")?;
                self.write_type(out, module, ty)?;
                write!(out, ">()")?;
            }
            Expression::Compose { ty, ref components } => {
                self.write_constructor_expression(
                    out,
                    module,
                    ty,
                    components,
                    write_expression,
                    expression_type,
                )?;
            }
            Expression::Splat { size, value } => {
                let size = conv::vector_size_str(size);
                // TODO: emit explicit element type if explicit types requested
                write!(out, "{SHADER_LIB}::Vec{size}::splat(")?;
                write_expression(out, value)?;
                write!(out, ")")?;
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Examine the type to write an appropriate constructor or literal expression for it.
    ///
    /// We do not delegate to a library trait for this because the construction
    /// must be const-compatible.
    fn write_constructor_expression<'a>(
        &self,
        out: &mut dyn Write,
        module: &Module,
        ty: Handle<naga::Type>,
        components: &[Handle<Expression>],
        write_expression: impl Copy + Fn(&mut dyn Write, Handle<Expression>) -> BackendResult,
        expression_type: impl Copy + Fn(Handle<Expression>) -> &'a TypeInner,
    ) -> BackendResult {
        use naga::VectorSize::{Bi, Quad, Tri};

        let ctor_name = match module.types[ty].inner {
            TypeInner::Vector { size, scalar: _ } => {
                // Vectors may be constructed by a collection of scalars and vectors which in
                // total have the required component count.

                let arg_sizes: ArrayVec<u8, 4> = components
                    .iter()
                    .map(|&component_expr| match *expression_type(component_expr) {
                        TypeInner::Scalar(_) => 1,
                        TypeInner::Vector { size, .. } => size as u8,
                        ref t => unreachable!(
                            "vector constructor argument should be a scalar or vector, not {t:?}"
                        ),
                    })
                    .collect();

                match (size, &*arg_sizes) {
                    (Bi, [1, 1]) => "new",
                    (Bi, [2]) => "from",
                    (Tri, [1, 1, 1]) => "new",
                    (Tri, [1, 2]) => "new_12",
                    (Tri, [2, 1]) => "new_21",
                    (Quad, [1, 1, 1, 1]) => "new",
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

            // Fallback: Assume that a suitable `T::new()` associated function
            // exists.
            _ => "new",
        };

        write!(out, "<")?;
        self.write_type(out, module, ty)?;
        write!(out, ">::{ctor_name}(")?;
        for (index, component) in components.iter().enumerate() {
            if index > 0 {
                write!(out, ", ")?;
            }
            write_expression(out, *component)?;
        }
        write!(out, ")")?;

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
    /// TODO: explain the indirection parameter of *this* function.
    fn write_expr_plain_form(
        &self,
        out: &mut dyn Write,
        module: &Module,
        info: &ModuleInfo,
        expr: Handle<Expression>,
        func_ctx: &back::FunctionCtx<'_>,
        indirection: Indirection,
    ) -> BackendResult {
        if let Some(name) = self.named_expressions.get(&expr) {
            write!(out, "{name}")?;
            return Ok(());
        }

        let expression = &func_ctx.expressions[expr];

        match *expression {
            Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::ZeroValue(_)
            | Expression::Compose { .. }
            | Expression::Splat { .. } => {
                self.write_possibly_const_expression(
                    out,
                    module,
                    info,
                    expr,
                    func_ctx.expressions,
                    |out, expr| self.write_expr(out, module, info, expr, func_ctx),
                    |expr| func_ctx.resolve_type(expr, &module.types),
                )?;
            }
            Expression::Override(_) => unreachable!(),
            Expression::FunctionArgument(pos) => {
                let name_key = func_ctx.argument_key(pos);
                let name = &self.names[&name_key];
                write!(out, "{name}")?;
            }
            Expression::Binary { op, left, right } => {
                let inputs_are_scalar = matches!(
                    *func_ctx.resolve_type(left, &module.types),
                    TypeInner::Scalar(_)
                ) && matches!(
                    *func_ctx.resolve_type(right, &module.types),
                    TypeInner::Scalar(_)
                );
                match (inputs_are_scalar, BinOpClassified::from(op)) {
                    (true, BinOpClassified::ScalarBool(_))
                    | (_, BinOpClassified::Vectorizable(_)) => {
                        write!(out, "(")?;
                        self.write_expr(out, module, info, left, func_ctx)?;
                        // TODO: Review whether any Rust operator semantics are incorrect
                        //  for shader code — if so, stop using `binary_operation_str`.
                        write!(out, " {} ", back::binary_operation_str(op))?;
                        self.write_expr(out, module, info, right, func_ctx)?;
                        write!(out, ")")?;
                    }
                    (_, BinOpClassified::ScalarBool(bop)) => {
                        self.write_expr(out, module, info, left, func_ctx)?;
                        write!(out, ".{}(", bop.to_vector_method())?;
                        self.write_expr(out, module, info, right, func_ctx)?;
                        write!(out, ")")?;
                    }
                }
            }
            Expression::Access { base, index } => {
                self.write_expr_with_indirection(out, module, info, base, func_ctx, indirection)?;
                write!(out, "[")?;
                self.write_expr(out, module, info, index, func_ctx)?;
                write!(out, " as usize]")?
            }
            Expression::AccessIndex { base, index } => {
                let base_ty_res = &func_ctx.info[base].ty;
                let mut resolved = base_ty_res.inner_with(&module.types);

                self.write_expr_with_indirection(out, module, info, base, func_ctx, indirection)?;

                let base_ty_handle = match *resolved {
                    TypeInner::Pointer { base, space: _ } => {
                        resolved = &module.types[base].inner;
                        Some(base)
                    }
                    _ => base_ty_res.handle(),
                };

                match *resolved {
                    TypeInner::Vector { .. } => {
                        // Write vector access as a swizzle
                        write!(out, ".{}", back::COMPONENTS[index as usize])?
                    }
                    TypeInner::Matrix { .. }
                    | TypeInner::Array { .. }
                    | TypeInner::BindingArray { .. }
                    | TypeInner::ValuePointer { .. } => write!(out, "[{index} as usize]")?,
                    TypeInner::Struct { .. } => {
                        // This will never panic in case the type is a `Struct`, this is not true
                        // for other types so we can only check while inside this match arm
                        let ty = base_ty_handle.unwrap();

                        write!(out, ".{}", &self.names[&NameKey::StructMember(ty, index)])?
                    }
                    ref other => unreachable!("cannot index into a {other:?}"),
                }
            }
            Expression::ImageSample { .. } => {
                return Err(Error::TexturesAreUnsupported {
                    found: "textureSample",
                });
            }
            Expression::ImageQuery { .. } => {
                return Err(Error::TexturesAreUnsupported {
                    found: "texture queries",
                });
            }
            Expression::ImageLoad { .. } => {
                return Err(Error::TexturesAreUnsupported {
                    found: "textureLoad",
                });
            }
            Expression::GlobalVariable(handle) => {
                let name = &self.names[&NameKey::GlobalVariable(handle)];
                write!(out, "self.{name}")?;
            }

            Expression::As {
                expr,
                kind: to_kind,
                convert: to_width,
            } => {
                use naga::TypeInner as Ti;

                let input_type = func_ctx.resolve_type(expr, &module.types);

                self.write_expr(out, module, info, expr, func_ctx)?;
                match (input_type, to_kind, to_width) {
                    (&Ti::Vector { size: _, scalar: _ }, to_kind, Some(to_width)) => {
                        // Call a glam vector cast method
                        write!(
                            out,
                            ".cast_elem_as_{elem_ty}()",
                            elem_ty = unwrap_to_rust(Scalar {
                                kind: to_kind,
                                width: to_width
                            }),
                        )?;
                    }
                    (&Ti::Scalar(_), to_kind, Some(to_width)) => {
                        // Coerce scalars using Rust 'as'
                        // TODO: replace Rust scalars with an explicit rt::Scalar type
                        write!(
                            out,
                            " as {}",
                            unwrap_to_rust(Scalar {
                                kind: to_kind,
                                width: to_width,
                            })
                        )?;
                    }
                    _ => {
                        // Unhandled case, produce debugging info
                        write!(
                            out,
                            " as _/* cast {input_type:?} to kind {to_kind:?} width {to_width:?} */"
                        )?;
                    }
                }
            }
            Expression::Load { pointer } => {
                self.write_expr_with_indirection(
                    out,
                    module,
                    info,
                    pointer,
                    func_ctx,
                    Indirection::Place,
                )?;
            }
            Expression::LocalVariable(handle) => {
                write!(out, "{}", self.names[&func_ctx.name_key(handle)])?
            }
            Expression::ArrayLength(expr) => {
                self.write_expr(out, module, info, expr, func_ctx)?;
                write!(out, ".len()")?;
            }

            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                self.write_expr(out, module, info, arg, func_ctx)?;
                write!(
                    out,
                    ".{method}(",
                    method = conv::math_function_to_method(fun)
                )?;
                for arg in [arg1, arg2, arg3].into_iter().flatten() {
                    self.write_expr(out, module, info, arg, func_ctx)?;
                    write!(out, ", ")?;
                }
                write!(out, ")")?
            }

            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                self.write_expr(out, module, info, vector, func_ctx)?;
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

                write!(out, "{unary}(")?;
                self.write_expr(out, module, info, expr, func_ctx)?;

                write!(out, ")")?
            }

            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let suffix = match *func_ctx.resolve_type(condition, &module.types) {
                    TypeInner::Scalar(Scalar::BOOL) => "",
                    TypeInner::Vector {
                        size,
                        scalar: Scalar::BOOL,
                    } => conv::vector_size_str(size),
                    _ => unreachable!("validation should have rejected this"),
                };
                write!(out, "{SHADER_LIB}::select{suffix}(")?;
                self.write_expr(out, module, info, reject, func_ctx)?;
                write!(out, ", ")?;
                self.write_expr(out, module, info, accept, func_ctx)?;
                write!(out, ", ")?;
                self.write_expr(out, module, info, condition, func_ctx)?;
                write!(out, ")")?
            }
            Expression::Derivative { axis, ctrl, expr } => {
                use naga::{DerivativeAxis as Axis, DerivativeControl as Ctrl};
                let op = match (axis, ctrl) {
                    (Axis::X, Ctrl::Coarse) => "dpdxCoarse",
                    (Axis::X, Ctrl::Fine) => "dpdxFine",
                    (Axis::X, Ctrl::None) => "dpdx",
                    (Axis::Y, Ctrl::Coarse) => "dpdyCoarse",
                    (Axis::Y, Ctrl::Fine) => "dpdyFine",
                    (Axis::Y, Ctrl::None) => "dpdy",
                    (Axis::Width, Ctrl::Coarse) => "fwidthCoarse",
                    (Axis::Width, Ctrl::Fine) => "fwidthFine",
                    (Axis::Width, Ctrl::None) => "fwidth",
                };
                write!(out, "{SHADER_LIB}::{op}(")?;
                self.write_expr(out, module, info, expr, func_ctx)?;
                write!(out, ")")?
            }
            Expression::Relational { fun, argument } => {
                use naga::RelationalFunction as Rf;

                let fun_name = match fun {
                    Rf::All => "all",
                    Rf::Any => "any",
                    Rf::IsNan => "is_nan",
                    Rf::IsInf => "is_inf",
                };
                write!(out, "{SHADER_LIB}::{fun_name}(")?;
                self.write_expr(out, module, info, argument, func_ctx)?;
                write!(out, ")")?
            }
            // Not supported yet
            Expression::RayQueryGetIntersection { .. } => unreachable!(),
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

    pub(super) fn write_type(
        &self,
        out: &mut dyn Write,
        module: &Module,
        handle: Handle<naga::Type>,
    ) -> BackendResult {
        let ty = &module.types[handle];
        match ty.inner {
            TypeInner::Struct { .. } => {
                out.write_str(self.names[&NameKey::Type(handle)].as_str())?
            }
            ref other => self.write_type_inner(out, module, other)?,
        }

        Ok(())
    }

    fn write_type_inner(
        &self,
        out: &mut dyn Write,
        module: &Module,
        inner: &TypeInner,
    ) -> BackendResult {
        match *inner {
            TypeInner::Vector { size, scalar } => write!(
                out,
                "{SHADER_LIB}::Vec{}<{}>",
                conv::vector_size_str(size),
                unwrap_to_rust(scalar),
            )?,
            TypeInner::Sampler { comparison: false } => {
                write!(out, "{SHADER_LIB}::Sampler")?;
            }
            TypeInner::Sampler { comparison: true } => {
                write!(out, "{SHADER_LIB}::SamplerComparison")?;
            }
            TypeInner::Image { .. } => {
                write!(out, "{SHADER_LIB}::Image")?;
            }
            TypeInner::Scalar(scalar) => {
                write!(out, "{}", unwrap_to_rust(scalar))?;
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
                match size {
                    naga::ArraySize::Constant(len) => {
                        self.write_type(out, module, base)?;
                        write!(out, "; {len}")?;
                    }
                    naga::ArraySize::Pending(..) => {
                        return Err(Error::Unimplemented("override array size".into()));
                    }
                    naga::ArraySize::Dynamic => {
                        self.write_type(out, module, base)?;
                    }
                }
                write!(out, "]")?;
            }
            TypeInner::BindingArray { .. } => {}
            TypeInner::Matrix { .. } => {
                return Err(Error::Unimplemented("matrices".into()));
            }
            TypeInner::Pointer { base, space: _ } => {
                if self.config.flags.contains(WriterFlags::RAW_POINTERS) {
                    write!(out, "*mut ")?;
                } else {
                    write!(out, "&mut ")?;
                }
                self.write_type(out, module, base)?;
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
            name: _,    // renamed instead
            space: _,   // no address spaces exist
            binding: _, // don't (yet) expose numeric binding locations
            ty,
            init: _, // TODO: need to put initializes in a newp() fn
        } = global;

        // Note bindings.
        // These are not emitted as attributes because Rust does not allow macro attributes to be
        // placed on struct fields.
        if let Some(naga::ResourceBinding { group, binding }) = global.binding {
            writeln!(out, "{INDENT}// group({group}) binding({binding})")?;
        }

        write!(
            out,
            "{INDENT}{}: ",
            &self.names[&NameKey::GlobalVariable(handle)]
        )?;
        self.write_type(out, module, ty)?;
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
        write!(
            out,
            "{INDENT}{INDENT}{}: ",
            &self.names[&NameKey::GlobalVariable(handle)]
        )?;

        if let Some(init) = global.init {
            self.write_const_expression(out, module, info, init)?;
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
        self.write_type(out, module, module.constants[handle].ty)?;
        write!(out, " = ")?;
        self.write_const_expression(out, module, info, init)?;
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
