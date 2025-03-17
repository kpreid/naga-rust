use alloc::{
    format,
    string::{String, ToString},
    vec,
};
use core::fmt::Write;

use naga::{
    Expression, Handle, Module, Scalar, ShaderStage, TypeInner,
    back::{self},
    proc::{self, ExpressionKindTracker, NameKey},
    valid,
};

use crate::util::{Baked, LevelNext};
use crate::{
    Error,
    conv::{self, BinOpClassified, KEYWORDS_2024, SHADER_LIB, ToRust, unwrap_to_rust},
};

// -------------------------------------------------------------------------------------------------

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

/// WGSL [attribute](https://gpuweb.github.io/gpuweb/wgsl/#attributes)
#[allow(dead_code, reason = "TODO: get rid of this WGSLism")]
enum Attribute {
    Binding(u32),
    BuiltIn(naga::BuiltIn),
    Group(u32),
    Invariant,
    Interpolate(Option<naga::Interpolation>, Option<naga::Sampling>),
    Location(u32),
    BlendSrc(u32),
    Stage(ShaderStage),
    WorkGroupSize([u32; 3]),
}

/// The WGSL form that `write_expr_with_indirection` should use to render a Naga
/// expression.
///
/// Sometimes a Naga `Expression` alone doesn't provide enough information to
/// choose the right rendering for it in WGSL. For example, one natural WGSL
/// rendering of a Naga `LocalVariable(x)` expression might be `&x`, since
/// `LocalVariable` produces a pointer to the local variable's storage. But when
/// rendering a `Store` statement, the `pointer` operand must be the left hand
/// side of a WGSL assignment, so the proper rendering is `x`.
///
/// The caller of `write_expr_with_indirection` must provide an `Expected` value
/// to indicate how ambiguous expressions should be rendered.
#[derive(Clone, Copy, Debug)]
enum Indirection {
    /// Render pointer-construction expressions as WGSL `ptr`-typed expressions.
    ///
    /// This is the right choice for most cases. Whenever a Naga pointer
    /// expression is not the `pointer` operand of a `Load` or `Store`, it
    /// must be a WGSL pointer expression.
    Ordinary,

    /// Render pointer-construction expressions as WGSL reference-typed
    /// expressions.
    ///
    /// For example, this is the right choice for the `pointer` operand when
    /// rendering a `Store` statement as a WGSL assignment.
    Reference,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct WriterFlags: u32 {
        /// Always annotate the type information instead of inferring.
        const EXPLICIT_TYPES = 0x1;

        /// Generate code using raw pointers instead of references.
        /// The resulting code is `unsafe` and may be unsound if the WGSL uses invalid pointers.
        const RAW_POINTERS = 0x2;

        /// Generate items with `pub` visibility instead of private.
        const PUBLIC = 0x3;
    }
}

/// Edition of Rust code to generate.
///
/// We currently only support one edition, but this exists anyway to prepare to document
/// any edition dependencies in the code generator.
enum Edition {
    Rust2024,
}

pub struct Writer<W> {
    out: W,
    flags: WriterFlags,
    #[allow(dead_code)]
    edition: Edition,
    names: naga::FastHashMap<NameKey, String>,
    namer: proc::Namer,
    named_expressions: naga::FastIndexMap<Handle<Expression>, String>,
    //required_polyfills: naga::FastIndexSet<InversePolyfill>,
}

impl<W: Write> Writer<W> {
    pub fn new(out: W, flags: WriterFlags) -> Self {
        Writer {
            out,
            flags,
            edition: Edition::Rust2024,
            names: naga::FastHashMap::default(),
            namer: proc::Namer::default(),
            named_expressions: naga::FastIndexMap::default(),
            //required_polyfills: naga::FastIndexSet::default(),
        }
    }

    fn reset(&mut self, module: &Module) {
        let Self {
            out: _,
            flags: _,
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
        named_expressions.clear();
        //self.required_polyfills.clear();
    }

    fn is_builtin_wgsl_struct(&self, module: &Module, handle: Handle<naga::Type>) -> bool {
        module
            .special_types
            .predeclared_types
            .values()
            .any(|t| *t == handle)
    }

    pub fn write(&mut self, module: &Module, info: &valid::ModuleInfo) -> BackendResult {
        if !module.overrides.is_empty() {
            return Err(Error::Unimplemented(
                "Pipeline constants are not yet supported for this back-end".to_string(),
            ));
        }

        self.reset(module);

        // Write top-level attributes
        write!(
            self.out,
            "\
                #[allow(unused)]\n\
                use ::naga_rust_rt::{{self, New as _, swizzles::{{Vec2Swizzles as _, Vec3Swizzles as _, Vec4Swizzles as _}}}};\n\
            "
        )?;

        // Write all structs
        for (handle, ty) in module.types.iter() {
            if let TypeInner::Struct { ref members, .. } = ty.inner {
                {
                    if !self.is_builtin_wgsl_struct(module, handle) {
                        self.write_struct(module, handle, members)?;
                        writeln!(self.out)?;
                    }
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
            self.write_global_constant(module, handle)?;
            // Add extra newline for readability on last iteration
            if constants.peek().is_none() {
                writeln!(self.out)?;
            }
        }

        if !module.global_variables.is_empty() {
            writeln!(self.out, "struct Globals {{")?;
            // TODO: we are going to need to sort out global variables into whether
            // they are fields or cons
            for (ty, global) in module.global_variables.iter() {
                self.write_global_variable_as_field(module, global, ty)?;
            }
            writeln!(self.out, "}}")?;
        }

        // Write all regular functions
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
            self.write_function(module, function, &func_ctx)?;

            writeln!(self.out)?;
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

            self.write_attributes(&attributes)?;

            let func_ctx = back::FunctionCtx {
                ty: back::FunctionType::EntryPoint(index as u16),
                info: info.get_entry_point(index),
                expressions: &ep.function.expressions,
                named_expressions: &ep.function.named_expressions,
                expr_kind_tracker: ExpressionKindTracker::from_arena(&ep.function.expressions),
            };
            self.write_function(module, &ep.function, &func_ctx)?;

            if index < module.entry_points.len() - 1 {
                writeln!(self.out)?;
            }
        }

        // Write any polyfills that were required.
        // for polyfill in &self.required_polyfills {
        //     writeln!(self.out)?;
        //     write!(self.out, "{}", polyfill.source)?;
        //     writeln!(self.out)?;
        // }

        Ok(())
    }

    /// Helper method used to write
    /// [functions](https://gpuweb.github.io/gpuweb/wgsl/#functions)
    ///
    /// # Notes
    /// Ends in a newline
    fn write_function(
        &mut self,
        module: &Module,
        func: &naga::Function,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        let func_name = match func_ctx.ty {
            back::FunctionType::EntryPoint(index) => &self.names[&NameKey::EntryPoint(index)],
            back::FunctionType::Function(handle) => &self.names[&NameKey::Function(handle)],
        };

        // Write function name
        let visibility = self.visibility();
        write!(
            self.out,
            "#[allow(unused, clippy::all)]\n\
            {visibility}fn {func_name}("
        )?;

        // Write function arguments
        for (index, arg) in func.arguments.iter().enumerate() {
            // // Write argument attribute if a binding is present
            // if let Some(ref binding) = arg.binding {
            //     self.write_attributes(&map_binding_to_attribute(binding))?;
            // }
            // Write argument name
            let argument_name = &self.names[&func_ctx.argument_key(index as u32)];

            write!(self.out, "{argument_name}: ")?;
            // Write argument type
            self.write_type(module, arg.ty)?;
            if index < func.arguments.len() - 1 {
                // Add a separator between args
                write!(self.out, ", ")?;
            }
        }

        write!(self.out, ")")?;

        // Write function return type
        if let Some(ref result) = func.result {
            write!(self.out, " -> ")?;
            // if let Some(ref binding) = result.binding {
            //     self.write_attributes(&map_binding_to_attribute(binding))?;
            // }
            self.write_type(module, result.ty)?;
        }

        write!(self.out, " {{")?;
        writeln!(self.out)?;

        // Write function local variables
        for (handle, local) in func.local_variables.iter() {
            // Write indentation (only for readability)
            write!(self.out, "{}", back::INDENT)?;

            // Write the local name
            // The leading space is important
            write!(
                self.out,
                "let mut {}: ",
                self.names[&func_ctx.name_key(handle)]
            )?;

            // Write the local type
            self.write_type(module, local.ty)?;

            // Write the local initializer if needed
            if let Some(init) = local.init {
                // Put the equal signal only if there's a initializer
                // The leading and trailing spaces aren't needed but help with readability
                write!(self.out, " = ")?;

                // Write the constant
                // `write_constant` adds no trailing or leading space/newline
                self.write_expr(module, init, func_ctx)?;
            }

            // Finish the local with `;` and add a newline (only for readability)
            writeln!(self.out, ";")?
        }

        if !func.local_variables.is_empty() {
            writeln!(self.out)?;
        }

        // Write the function body (statement list)
        for sta in func.body.iter() {
            // The indentation should always be 1 when writing the function body
            self.write_stmt(module, sta, func_ctx, back::Level(1))?;
        }

        writeln!(self.out, "}}")?;

        self.named_expressions.clear();

        Ok(())
    }

    /// Helper method to write a attribute
    fn write_attributes(&mut self, attributes: &[Attribute]) -> BackendResult {
        for attribute in attributes {
            match *attribute {
                Attribute::Location(id) => write!(self.out, "@location({id}) ")?,
                Attribute::BlendSrc(blend_src) => write!(self.out, "@blend_src({blend_src}) ")?,
                Attribute::BuiltIn(_builtin_attrib) => {
                    // let builtin = builtin_attrib.to_wgsl_if_implemented()?;
                    // write!(self.out, "@builtin({builtin}) ")?;
                }
                Attribute::Stage(shader_stage) => {
                    let stage_str = match shader_stage {
                        ShaderStage::Vertex => "vertex",
                        ShaderStage::Fragment => "fragment",
                        ShaderStage::Compute => "compute",
                    };
                    writeln!(self.out, "#[{SHADER_LIB}::{stage_str}]")?;
                }
                Attribute::WorkGroupSize(size) => {
                    writeln!(
                        self.out,
                        "#[{SHADER_LIB}::workgroup_size({}, {}, {})]",
                        size[0], size[1], size[2]
                    )?;
                }
                Attribute::Binding(id) => writeln!(self.out, "#[{SHADER_LIB}::binding({id})]")?,
                Attribute::Group(id) => writeln!(self.out, "#[{SHADER_LIB}::group({id})]")?,
                Attribute::Invariant => writeln!(self.out, "#[{SHADER_LIB}::invariant]")?,
                Attribute::Interpolate(interpolation, sampling) => {
                    if sampling.is_some() && sampling != Some(naga::Sampling::Center) {
                        let interpolation = interpolation
                            .unwrap_or(naga::Interpolation::Perspective)
                            .to_rust();
                        let sampling = sampling.unwrap_or(naga::Sampling::Center).to_rust();
                        writeln!(
                            self.out,
                            "#[{SHADER_LIB}::interpolate({interpolation}, {sampling})]"
                        )?;
                    } else if interpolation.is_some()
                        && interpolation != Some(naga::Interpolation::Perspective)
                    {
                        let interpolation = interpolation
                            .unwrap_or(naga::Interpolation::Perspective)
                            .to_rust();
                        writeln!(self.out, "#[{SHADER_LIB}::interpolate({interpolation})]")?;
                    }
                }
            };
        }
        Ok(())
    }

    /// Helper method used to write structs
    /// Write the full declaration of a struct type.
    ///
    /// Write out a definition of the struct type referred to by
    /// `handle` in `module`. The output will be an instance of the
    /// `struct_decl` production in the WGSL grammar.
    ///
    /// Use `members` as the list of `handle`'s members. (This
    /// function is usually called after matching a `TypeInner`, so
    /// the callers already have the members at hand.)
    fn write_struct(
        &mut self,
        module: &Module,
        handle: Handle<naga::Type>,
        members: &[naga::StructMember],
    ) -> BackendResult {
        let visibility = self.visibility();
        write!(
            self.out,
            "{visibility}struct {}",
            self.names[&NameKey::Type(handle)]
        )?;
        write!(self.out, " {{")?;
        writeln!(self.out)?;
        for (index, member) in members.iter().enumerate() {
            // The indentation is only for readability
            write!(self.out, "{}", back::INDENT)?;
            // if let Some(ref binding) = member.binding {
            //     self.write_attributes(&map_binding_to_attribute(binding))?;
            // }
            // Write struct member name and type
            let member_name = &self.names[&NameKey::StructMember(handle, index as u32)];
            write!(self.out, "pub {member_name}: ")?;
            self.write_type(module, member.ty)?;
            write!(self.out, ",")?;
            writeln!(self.out)?;
        }

        writeln!(self.out, "}}")?;

        Ok(())
    }

    /// Helper method used to write statements
    ///
    /// # Notes
    /// Always adds a newline
    fn write_stmt(
        &mut self,
        module: &Module,
        stmt: &naga::Statement,
        func_ctx: &back::FunctionCtx<'_>,
        level: back::Level,
    ) -> BackendResult {
        use naga::{Expression, Statement};

        match *stmt {
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let info = &func_ctx.info[handle];
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
                        if min_ref_count <= info.ref_count || required_baking_expr {
                            Some(Baked(handle).to_string())
                        } else {
                            None
                        }
                    };

                    if let Some(name) = expr_name {
                        write!(self.out, "{level}")?;
                        self.start_named_expr(module, handle, func_ctx, &name)?;
                        self.write_expr(module, handle, func_ctx)?;
                        self.named_expressions.insert(handle, name);
                        writeln!(self.out, ";")?;
                    }
                }
            }
            // TODO: copy-paste from glsl-out
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                write!(self.out, "{level}")?;
                write!(self.out, "if ")?;
                self.write_expr(module, condition, func_ctx)?;
                writeln!(self.out, " {{")?;

                let l2 = level.next();
                for sta in accept {
                    // Increase indentation to help with readability
                    self.write_stmt(module, sta, func_ctx, l2)?;
                }

                // If there are no statements in the reject block we skip writing it
                // This is only for readability
                if !reject.is_empty() {
                    writeln!(self.out, "{level}}} else {{")?;

                    for sta in reject {
                        // Increase indentation to help with readability
                        self.write_stmt(module, sta, func_ctx, l2)?;
                    }
                }

                writeln!(self.out, "{level}}}")?
            }
            Statement::Return { value } => {
                write!(self.out, "{level}")?;
                write!(self.out, "return")?;
                if let Some(return_value) = value {
                    // The leading space is important
                    write!(self.out, " ")?;
                    self.write_expr(module, return_value, func_ctx)?;
                }
                writeln!(self.out, ";")?;
            }
            // TODO: copy-paste from glsl-out
            Statement::Kill => {
                write!(self.out, "{level}")?;
                writeln!(self.out, "discard;")?
            }
            Statement::Store { pointer, value } => {
                write!(self.out, "{level}")?;

                let is_atomic_pointer = func_ctx
                    .resolve_type(pointer, &module.types)
                    .is_atomic_pointer(&module.types);

                if is_atomic_pointer {
                    write!(self.out, "atomicStore(")?;
                    self.write_expr(module, pointer, func_ctx)?;
                    write!(self.out, ", ")?;
                    self.write_expr(module, value, func_ctx)?;
                    write!(self.out, ")")?;
                } else {
                    self.write_expr_with_indirection(
                        module,
                        pointer,
                        func_ctx,
                        Indirection::Reference,
                    )?;
                    write!(self.out, " = ")?;
                    self.write_expr(module, value, func_ctx)?;
                }
                writeln!(self.out, ";")?
            }
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                write!(self.out, "{level}")?;
                if let Some(expr) = result {
                    let name = Baked(expr).to_string();
                    self.start_named_expr(module, expr, func_ctx, &name)?;
                    self.named_expressions.insert(expr, name);
                }
                let func_name = &self.names[&NameKey::Function(function)];
                write!(self.out, "{func_name}(")?;
                for (index, &argument) in arguments.iter().enumerate() {
                    if index != 0 {
                        write!(self.out, ", ")?;
                    }
                    self.write_expr(module, argument, func_ctx)?;
                }
                writeln!(self.out, ");")?
            }
            Statement::Atomic { .. } => {
                todo!("Statement::Atomic");
            }
            Statement::ImageAtomic { .. } => {
                todo!("Statement::ImageAtomic");
            }
            Statement::WorkGroupUniformLoad { .. } => {
                todo!("Statement::WorkGroupUniformLoad");
            }
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                write!(self.out, "{level}")?;
                write!(self.out, "textureStore(")?;
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;
                if let Some(array_index_expr) = array_index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, array_index_expr, func_ctx)?;
                }
                write!(self.out, ", ")?;
                self.write_expr(module, value, func_ctx)?;
                writeln!(self.out, ");")?;
            }
            // TODO: copy-paste from glsl-out
            Statement::Block(ref block) => {
                write!(self.out, "{level}")?;
                writeln!(self.out, "{{")?;
                for sta in block.iter() {
                    // Increase the indentation to help with readability
                    self.write_stmt(module, sta, func_ctx, level.next())?
                }
                writeln!(self.out, "{level}}}")?
            }
            Statement::Switch {
                selector,
                ref cases,
            } => {
                // Start the switch
                write!(self.out, "{level}")?;
                write!(self.out, "switch ")?;
                self.write_expr(module, selector, func_ctx)?;
                writeln!(self.out, " {{")?;

                let l2 = level.next();
                let mut new_case = true;
                for case in cases {
                    if case.fall_through && !case.body.is_empty() {
                        // TODO: we could do the same workaround as we did for the HLSL backend
                        return Err(Error::Unimplemented(
                            "fall-through switch case block".into(),
                        ));
                    }

                    match case.value {
                        naga::SwitchValue::I32(value) => {
                            if new_case {
                                write!(self.out, "{l2}case ")?;
                            }
                            write!(self.out, "{value}")?;
                        }
                        naga::SwitchValue::U32(value) => {
                            if new_case {
                                write!(self.out, "{l2}case ")?;
                            }
                            write!(self.out, "{value}u")?;
                        }
                        naga::SwitchValue::Default => {
                            if new_case {
                                if case.fall_through {
                                    write!(self.out, "{l2}case ")?;
                                } else {
                                    write!(self.out, "{l2}")?;
                                }
                            }
                            write!(self.out, "default")?;
                        }
                    }

                    new_case = !case.fall_through;

                    if case.fall_through {
                        write!(self.out, ", ")?;
                    } else {
                        writeln!(self.out, ": {{")?;
                    }

                    for sta in case.body.iter() {
                        self.write_stmt(module, sta, func_ctx, l2.next())?;
                    }

                    if !case.fall_through {
                        writeln!(self.out, "{l2}}}")?;
                    }
                }

                writeln!(self.out, "{level}}}")?
            }
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                write!(self.out, "{level}")?;
                writeln!(self.out, "loop {{")?;

                let l2 = level.next();
                for sta in body.iter() {
                    self.write_stmt(module, sta, func_ctx, l2)?;
                }

                // The continuing is optional so we don't need to write it if
                // it is empty, but the `break if` counts as a continuing statement
                // so even if `continuing` is empty we must generate it if a
                // `break if` exists
                if !continuing.is_empty() || break_if.is_some() {
                    writeln!(self.out, "{l2}continuing {{")?;
                    for sta in continuing.iter() {
                        self.write_stmt(module, sta, func_ctx, l2.next())?;
                    }

                    // The `break if` is always the last
                    // statement of the `continuing` block
                    if let Some(condition) = break_if {
                        // The trailing space is important
                        write!(self.out, "{}break if ", l2.next())?;
                        self.write_expr(module, condition, func_ctx)?;
                        // Close the `break if` statement
                        writeln!(self.out, ";")?;
                    }

                    writeln!(self.out, "{l2}}}")?;
                }

                writeln!(self.out, "{level}}}")?
            }
            Statement::Break => {
                writeln!(self.out, "{level}break;")?;
            }
            Statement::Continue => {
                writeln!(self.out, "{level}continue;")?;
            }
            Statement::Barrier(barrier) => {
                if barrier.contains(naga::Barrier::STORAGE) {
                    writeln!(self.out, "{level}nstd::storage_barrier();")?;
                }

                if barrier.contains(naga::Barrier::WORK_GROUP) {
                    writeln!(self.out, "{level}nstd::workgroup_barrier();")?;
                }

                if barrier.contains(naga::Barrier::SUB_GROUP) {
                    writeln!(self.out, "{level}nstd::subgroup_barrier();")?;
                }

                // if barrier.contains(naga::Barrier::TEXTURE) {
                //     writeln!(self.out, "{level}nstd::texture_barrier();")?;
                // }

                // TODO: exhaustivity
            }
            Statement::RayQuery { .. } => unreachable!(),
            Statement::SubgroupBallot { result, predicate } => {
                write!(self.out, "{level}")?;
                let res_name = Baked(result).to_string();
                self.start_named_expr(module, result, func_ctx, &res_name)?;
                self.named_expressions.insert(result, res_name);

                write!(self.out, "subgroupBallot(")?;
                if let Some(predicate) = predicate {
                    self.write_expr(module, predicate, func_ctx)?;
                }
                writeln!(self.out, ");")?;
            }
            Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => {
                write!(self.out, "{level}")?;
                let res_name = Baked(result).to_string();
                self.start_named_expr(module, result, func_ctx, &res_name)?;
                self.named_expressions.insert(result, res_name);

                match (collective_op, op) {
                    (naga::CollectiveOperation::Reduce, naga::SubgroupOperation::All) => {
                        write!(self.out, "subgroupAll(")?
                    }
                    (naga::CollectiveOperation::Reduce, naga::SubgroupOperation::Any) => {
                        write!(self.out, "subgroupAny(")?
                    }
                    (naga::CollectiveOperation::Reduce, naga::SubgroupOperation::Add) => {
                        write!(self.out, "subgroupAdd(")?
                    }
                    (naga::CollectiveOperation::Reduce, naga::SubgroupOperation::Mul) => {
                        write!(self.out, "subgroupMul(")?
                    }
                    (naga::CollectiveOperation::Reduce, naga::SubgroupOperation::Max) => {
                        write!(self.out, "subgroupMax(")?
                    }
                    (naga::CollectiveOperation::Reduce, naga::SubgroupOperation::Min) => {
                        write!(self.out, "subgroupMin(")?
                    }
                    (naga::CollectiveOperation::Reduce, naga::SubgroupOperation::And) => {
                        write!(self.out, "subgroupAnd(")?
                    }
                    (naga::CollectiveOperation::Reduce, naga::SubgroupOperation::Or) => {
                        write!(self.out, "subgroupOr(")?
                    }
                    (naga::CollectiveOperation::Reduce, naga::SubgroupOperation::Xor) => {
                        write!(self.out, "subgroupXor(")?
                    }
                    (naga::CollectiveOperation::ExclusiveScan, naga::SubgroupOperation::Add) => {
                        write!(self.out, "subgroupExclusiveAdd(")?
                    }
                    (naga::CollectiveOperation::ExclusiveScan, naga::SubgroupOperation::Mul) => {
                        write!(self.out, "subgroupExclusiveMul(")?
                    }
                    (naga::CollectiveOperation::InclusiveScan, naga::SubgroupOperation::Add) => {
                        write!(self.out, "subgroupInclusiveAdd(")?
                    }
                    (naga::CollectiveOperation::InclusiveScan, naga::SubgroupOperation::Mul) => {
                        write!(self.out, "subgroupInclusiveMul(")?
                    }
                    _ => unimplemented!(),
                }
                self.write_expr(module, argument, func_ctx)?;
                writeln!(self.out, ");")?;
            }
            Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                write!(self.out, "{level}")?;
                let res_name = Baked(result).to_string();
                self.start_named_expr(module, result, func_ctx, &res_name)?;
                self.named_expressions.insert(result, res_name);

                match mode {
                    naga::GatherMode::BroadcastFirst => {
                        write!(self.out, "subgroupBroadcastFirst(")?;
                    }
                    naga::GatherMode::Broadcast(_) => {
                        write!(self.out, "subgroupBroadcast(")?;
                    }
                    naga::GatherMode::Shuffle(_) => {
                        write!(self.out, "subgroupShuffle(")?;
                    }
                    naga::GatherMode::ShuffleDown(_) => {
                        write!(self.out, "subgroupShuffleDown(")?;
                    }
                    naga::GatherMode::ShuffleUp(_) => {
                        write!(self.out, "subgroupShuffleUp(")?;
                    }
                    naga::GatherMode::ShuffleXor(_) => {
                        write!(self.out, "subgroupShuffleXor(")?;
                    }
                }
                self.write_expr(module, argument, func_ctx)?;
                match mode {
                    naga::GatherMode::BroadcastFirst => {}
                    naga::GatherMode::Broadcast(index)
                    | naga::GatherMode::Shuffle(index)
                    | naga::GatherMode::ShuffleDown(index)
                    | naga::GatherMode::ShuffleUp(index)
                    | naga::GatherMode::ShuffleXor(index) => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, index, func_ctx)?;
                    }
                }
                writeln!(self.out, ");")?;
            }
        }

        Ok(())
    }

    /// Return the sort of indirection that `expr`'s plain form evaluates to.
    ///
    /// An expression's 'plain form' is the most general rendition of that
    /// expression into WGSL, lacking `&` or `*` operators:
    ///
    /// - The plain form of `LocalVariable(x)` is simply `x`, which is a reference
    ///   to the local variable's storage.
    ///
    /// - The plain form of `GlobalVariable(g)` is simply `g`, which is usually a
    ///   reference to the global variable's storage. However, globals in the
    ///   `Handle` address space are immutable, and `GlobalVariable` expressions for
    ///   those produce the value directly, not a pointer to it. Such
    ///   `GlobalVariable` expressions are `Ordinary`.
    ///
    /// - `Access` and `AccessIndex` are `Reference` when their `base` operand is a
    ///   pointer. If they are applied directly to a composite value, they are
    ///   `Ordinary`.
    ///
    /// Note that `FunctionArgument` expressions are never `Reference`, even when
    /// the argument's type is `Pointer`. `FunctionArgument` always evaluates to the
    /// argument's value directly, so any pointer it produces is merely the value
    /// passed by the caller.
    fn plain_form_indirection(
        &self,
        expr: Handle<Expression>,
        module: &Module,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> Indirection {
        use naga::Expression as Ex;

        // Named expressions are `let` expressions, which apply the Load Rule,
        // so if their type is a Naga pointer, then that must be a WGSL pointer
        // as well.
        if self.named_expressions.contains_key(&expr) {
            return Indirection::Ordinary;
        }

        match func_ctx.expressions[expr] {
            Ex::LocalVariable(_) => Indirection::Reference,
            Ex::GlobalVariable(handle) => {
                let global = &module.global_variables[handle];
                match global.space {
                    naga::AddressSpace::Handle => Indirection::Ordinary,
                    _ => Indirection::Reference,
                }
            }
            Ex::Access { base, .. } | Ex::AccessIndex { base, .. } => {
                let base_ty = func_ctx.resolve_type(base, &module.types);
                match *base_ty {
                    TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => {
                        Indirection::Reference
                    }
                    _ => Indirection::Ordinary,
                }
            }
            _ => Indirection::Ordinary,
        }
    }

    fn start_named_expr(
        &mut self,
        module: &Module,
        handle: Handle<Expression>,
        func_ctx: &back::FunctionCtx,
        name: &str,
    ) -> BackendResult {
        // Write variable name
        write!(self.out, "let {name}")?;
        if self.flags.contains(WriterFlags::EXPLICIT_TYPES) {
            write!(self.out, ": ")?;
            let ty = &func_ctx.info[handle].ty;
            // Write variable type
            match *ty {
                proc::TypeResolution::Handle(handle) => {
                    self.write_type(module, handle)?;
                }
                proc::TypeResolution::Value(ref inner) => {
                    self.write_type_inner(module, inner)?;
                }
            }
        }

        write!(self.out, " = ")?;
        Ok(())
    }

    /// Write the ordinary Rust form of `expr`.
    ///
    /// See `write_expr_with_indirection` for details.
    fn write_expr(
        &mut self,
        module: &Module,
        expr: Handle<Expression>,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        self.write_expr_with_indirection(module, expr, func_ctx, Indirection::Ordinary)
    }

    /// Write `expr` as a WGSL expression with the requested indirection.
    ///
    /// In terms of the WGSL grammar, the resulting expression is a
    /// `singular_expression`. It may be parenthesized. This makes it suitable
    /// for use as the operand of a unary or binary operator without worrying
    /// about precedence.
    ///
    /// This does not produce newlines or indentation.
    ///
    /// The `requested` argument indicates (roughly) whether Naga
    /// `Pointer`-valued expressions represent WGSL references or pointers. See
    /// `Indirection` for details.
    fn write_expr_with_indirection(
        &mut self,
        module: &Module,
        expr: Handle<Expression>,
        func_ctx: &back::FunctionCtx<'_>,
        requested: Indirection,
    ) -> BackendResult {
        // If the plain form of the expression is not what we need, emit the
        // operator necessary to correct that.
        let plain = self.plain_form_indirection(expr, module, func_ctx);
        match (requested, plain) {
            (Indirection::Ordinary, Indirection::Reference) => {
                write!(self.out, "(&")?;
                self.write_expr_plain_form(module, expr, func_ctx, plain)?;
                write!(self.out, ")")?;
            }
            (Indirection::Reference, Indirection::Ordinary) => {
                write!(self.out, "(*")?;
                self.write_expr_plain_form(module, expr, func_ctx, plain)?;
                write!(self.out, ")")?;
            }
            (_, _) => self.write_expr_plain_form(module, expr, func_ctx, plain)?,
        }

        Ok(())
    }

    fn write_const_expression(
        &mut self,
        module: &Module,
        expr: Handle<Expression>,
        arena: &naga::Arena<Expression>,
    ) -> BackendResult {
        self.write_possibly_const_expression(module, expr, arena, |writer, expr| {
            writer.write_const_expression(module, expr, arena)
        })
    }

    fn write_possibly_const_expression<E>(
        &mut self,
        module: &Module,
        expr: Handle<Expression>,
        expressions: &naga::Arena<Expression>,
        write_expression: E,
    ) -> BackendResult
    where
        E: Copy + Fn(&mut Self, Handle<Expression>) -> BackendResult,
    {
        match expressions[expr] {
            Expression::Literal(literal) => match literal {
                naga::Literal::F32(value) => write!(self.out, "{value}f32")?,
                naga::Literal::U32(value) => write!(self.out, "{value}u32")?,
                naga::Literal::I32(value) => {
                    write!(self.out, "{value}i32")?;
                }
                naga::Literal::Bool(value) => write!(self.out, "{value}")?,
                naga::Literal::F64(value) => write!(self.out, "{value}f64")?,
                naga::Literal::I64(value) => {
                    write!(self.out, "{value}i64")?;
                }
                naga::Literal::U64(value) => write!(self.out, "{value}u64")?,
                naga::Literal::AbstractInt(_) | naga::Literal::AbstractFloat(_) => {
                    return Err(Error::Custom(
                        "Abstract types should not appear in IR presented to backends".into(),
                    ));
                }
            },
            Expression::Constant(handle) => {
                let constant = &module.constants[handle];
                if constant.name.is_some() {
                    write!(self.out, "{}", self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.write_const_expression(module, constant.init, &module.global_expressions)?;
                }
            }
            Expression::ZeroValue(ty) => {
                write!(self.out, "{SHADER_LIB}::zero::<")?;
                self.write_type(module, ty)?;
                write!(self.out, ">()")?;
            }
            Expression::Compose { ty, ref components } => {
                self.write_constructor_expression(module, ty, components, write_expression)?;
            }
            Expression::Splat { size, value } => {
                let size = conv::vector_size_str(size);
                write!(self.out, "{SHADER_LIB}::splat{size}(")?;
                write_expression(self, value)?;
                write!(self.out, ")")?;
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Examine the type to write an appropriate constructor or literal expression for it.
    ///
    /// We do not delegate to a library trait for this because the construction
    /// must be const-compatible.
    fn write_constructor_expression<E>(
        &mut self,
        module: &Module,
        ty: Handle<naga::Type>,
        components: &[Handle<Expression>],
        write_expression: E,
    ) -> BackendResult
    where
        E: Fn(&mut Self, Handle<Expression>) -> BackendResult,
    {
        write!(self.out, "<")?;
        self.write_type(module, ty)?;
        // TODO: use conventional Rust ctor syntax more often
        write!(self.out, ">::new(")?;
        for (index, component) in components.iter().enumerate() {
            if index > 0 {
                write!(self.out, ", ")?;
            }
            write_expression(self, *component)?;
        }
        write!(self.out, ")")?;

        Ok(())
    }

    /// Write the 'plain form' of `expr`.
    ///
    /// An expression's 'plain form' is the most general rendition of that
    /// expression into WGSL, lacking `&` or `*` operators. The plain forms of
    /// `LocalVariable(x)` and `GlobalVariable(g)` are simply `x` and `g`. Such
    /// Naga expressions represent both WGSL pointers and references; it's the
    /// caller's responsibility to distinguish those cases appropriately.
    fn write_expr_plain_form(
        &mut self,
        module: &Module,
        expr: Handle<Expression>,
        func_ctx: &back::FunctionCtx<'_>,
        indirection: Indirection,
    ) -> BackendResult {
        if let Some(name) = self.named_expressions.get(&expr) {
            write!(self.out, "{name}")?;
            return Ok(());
        }

        let expression = &func_ctx.expressions[expr];

        // Write the plain WGSL form of a Naga expression.
        //
        // The plain form of `LocalVariable` and `GlobalVariable` expressions is
        // simply the variable name; `*` and `&` operators are never emitted.
        //
        // The plain form of `Access` and `AccessIndex` expressions are WGSL
        // `postfix_expression` forms for member/component access and
        // subscripting.
        match *expression {
            Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::ZeroValue(_)
            | Expression::Compose { .. }
            | Expression::Splat { .. } => {
                self.write_possibly_const_expression(
                    module,
                    expr,
                    func_ctx.expressions,
                    |writer, expr| writer.write_expr(module, expr, func_ctx),
                )?;
            }
            Expression::Override(_) => unreachable!(),
            Expression::FunctionArgument(pos) => {
                let name_key = func_ctx.argument_key(pos);
                let name = &self.names[&name_key];
                write!(self.out, "{name}")?;
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
                        write!(self.out, "(")?;
                        self.write_expr(module, left, func_ctx)?;
                        // TODO: Review whether any Rust operator semantics are incorrect
                        //  for shader code — if so, stop using `binary_operation_str`.
                        write!(self.out, " {} ", back::binary_operation_str(op))?;
                        self.write_expr(module, right, func_ctx)?;
                        write!(self.out, ")")?;
                    }
                    (_, BinOpClassified::ScalarBool(bop)) => {
                        // TODO: generated function name is a placeholder
                        write!(self.out, "{SHADER_LIB}::{}(", bop.to_vector_fn())?;
                        self.write_expr(module, left, func_ctx)?;
                        write!(self.out, ", ")?;
                        self.write_expr(module, right, func_ctx)?;
                        write!(self.out, ")")?;
                    }
                }
            }
            Expression::Access { base, index } => {
                self.write_expr_with_indirection(module, base, func_ctx, indirection)?;
                write!(self.out, "[")?;
                self.write_expr(module, index, func_ctx)?;
                write!(self.out, " as usize]")?
            }
            Expression::AccessIndex { base, index } => {
                let base_ty_res = &func_ctx.info[base].ty;
                let mut resolved = base_ty_res.inner_with(&module.types);

                self.write_expr_with_indirection(module, base, func_ctx, indirection)?;

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
                        write!(self.out, ".{}", back::COMPONENTS[index as usize])?
                    }
                    TypeInner::Matrix { .. }
                    | TypeInner::Array { .. }
                    | TypeInner::BindingArray { .. }
                    | TypeInner::ValuePointer { .. } => write!(self.out, "[{index} as usize]")?,
                    TypeInner::Struct { .. } => {
                        // This will never panic in case the type is a `Struct`, this is not true
                        // for other types so we can only check while inside this match arm
                        let ty = base_ty_handle.unwrap();

                        write!(
                            self.out,
                            ".{}",
                            &self.names[&NameKey::StructMember(ty, index)]
                        )?
                    }
                    ref other => return Err(Error::Custom(format!("Cannot index {other:?}"))),
                }
            }
            Expression::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                use naga::SampleLevel as Sl;

                let suffix_cmp = match depth_ref {
                    Some(_) => "Compare",
                    None => "",
                };
                let suffix_level = match level {
                    Sl::Auto => "",
                    Sl::Zero | Sl::Exact(_) => "Level",
                    Sl::Bias(_) => "Bias",
                    Sl::Gradient { .. } => "Grad",
                };

                write!(self.out, "textureSample{suffix_cmp}{suffix_level}(")?;
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, sampler, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;

                if let Some(array_index) = array_index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, array_index, func_ctx)?;
                }

                if let Some(depth_ref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.write_expr(module, depth_ref, func_ctx)?;
                }

                match level {
                    Sl::Auto => {}
                    Sl::Zero => {
                        // Level 0 is implied for depth comparison
                        if depth_ref.is_none() {
                            write!(self.out, ", 0.0")?;
                        }
                    }
                    Sl::Exact(expr) => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, expr, func_ctx)?;
                    }
                    Sl::Bias(expr) => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, expr, func_ctx)?;
                    }
                    Sl::Gradient { x, y } => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, x, func_ctx)?;
                        write!(self.out, ", ")?;
                        self.write_expr(module, y, func_ctx)?;
                    }
                }

                if let Some(offset) = offset {
                    write!(self.out, ", ")?;
                    self.write_const_expression(module, offset, func_ctx.expressions)?;
                }

                write!(self.out, ")")?;
            }

            Expression::ImageSample {
                image,
                sampler,
                gather: Some(component),
                coordinate,
                array_index,
                offset,
                level: _,
                depth_ref,
            } => {
                let suffix_cmp = match depth_ref {
                    Some(_) => "Compare",
                    None => "",
                };

                write!(self.out, "textureGather{suffix_cmp}(")?;
                match *func_ctx.resolve_type(image, &module.types) {
                    TypeInner::Image {
                        class: naga::ImageClass::Depth { multi: _ },
                        ..
                    } => {}
                    _ => {
                        write!(self.out, "{}, ", component as u8)?;
                    }
                }
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, sampler, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;

                if let Some(array_index) = array_index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, array_index, func_ctx)?;
                }

                if let Some(depth_ref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.write_expr(module, depth_ref, func_ctx)?;
                }

                if let Some(offset) = offset {
                    write!(self.out, ", ")?;
                    self.write_const_expression(module, offset, func_ctx.expressions)?;
                }

                write!(self.out, ")")?;
            }
            Expression::ImageQuery { image, query } => {
                use naga::ImageQuery as Iq;

                let texture_function = match query {
                    Iq::Size { .. } => "textureDimensions",
                    Iq::NumLevels => "textureNumLevels",
                    Iq::NumLayers => "textureNumLayers",
                    Iq::NumSamples => "textureNumSamples",
                };

                write!(self.out, "{texture_function}(")?;
                self.write_expr(module, image, func_ctx)?;
                if let Iq::Size { level: Some(level) } = query {
                    write!(self.out, ", ")?;
                    self.write_expr(module, level, func_ctx)?;
                };
                write!(self.out, ")")?;
            }

            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                write!(self.out, "textureLoad(")?;
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;
                if let Some(array_index) = array_index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, array_index, func_ctx)?;
                }
                if let Some(index) = sample.or(level) {
                    write!(self.out, ", ")?;
                    self.write_expr(module, index, func_ctx)?;
                }
                write!(self.out, ")")?;
            }
            Expression::GlobalVariable(handle) => {
                let name = &self.names[&NameKey::GlobalVariable(handle)];
                write!(self.out, "{name}")?;
            }

            Expression::As {
                expr,
                kind: to_kind,
                convert: to_width,
            } => {
                use naga::TypeInner as Ti;

                let input_type = func_ctx.resolve_type(expr, &module.types);

                self.write_expr(module, expr, func_ctx)?;
                match (input_type, to_kind, to_width) {
                    (&Ti::Vector { size, scalar: _ }, to_kind, Some(to_width)) => {
                        // Call a glam vector cast method
                        write!(
                            self.out,
                            ".as_{prefix}vec{size}()",
                            prefix = conv::lower_glam_prefix(Scalar {
                                kind: to_kind,
                                width: to_width
                            }),
                            size = size as u8
                        )?;
                    }
                    (&Ti::Scalar(_), to_kind, Some(to_width)) => {
                        // Coerce scalars using Rust 'as'
                        write!(
                            self.out,
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
                            self.out,
                            " as _/* cast {input_type:?} to kind {to_kind:?} width {to_width:?} */"
                        )?;
                    }
                }
            }
            Expression::Load { pointer } => {
                self.write_expr_with_indirection(
                    module,
                    pointer,
                    func_ctx,
                    Indirection::Reference,
                )?;
            }
            Expression::LocalVariable(handle) => {
                write!(self.out, "{}", self.names[&func_ctx.name_key(handle)])?
            }
            Expression::ArrayLength(expr) => {
                write!(self.out, "arrayLength(")?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?;
            }

            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                write!(self.out, "{fun_name}(", fun_name = fun.to_rust())?;
                self.write_expr(module, arg, func_ctx)?;
                for arg in IntoIterator::into_iter([arg1, arg2, arg3]).flatten() {
                    write!(self.out, ", ")?;
                    self.write_expr(module, arg, func_ctx)?;
                }
                write!(self.out, ")")?
            }

            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                self.write_expr(module, vector, func_ctx)?;
                write!(self.out, ".")?;
                for &sc in pattern[..size as usize].iter() {
                    self.out.write_char(back::COMPONENTS[sc as usize])?;
                }
                write!(self.out, "()")?;
            }
            Expression::Unary { op, expr } => {
                let unary = match op {
                    naga::UnaryOperator::Negate => "-",
                    naga::UnaryOperator::LogicalNot => "!",
                    naga::UnaryOperator::BitwiseNot => "!",
                };

                write!(self.out, "{unary}(")?;
                self.write_expr(module, expr, func_ctx)?;

                write!(self.out, ")")?
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
                write!(self.out, "{SHADER_LIB}::select{suffix}(")?;
                self.write_expr(module, reject, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, accept, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, condition, func_ctx)?;
                write!(self.out, ")")?
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
                write!(self.out, "{SHADER_LIB}::{op}(")?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?
            }
            Expression::Relational { fun, argument } => {
                use naga::RelationalFunction as Rf;

                let fun_name = match fun {
                    Rf::All => "all",
                    Rf::Any => "any",
                    _ => return Err(Error::UnsupportedRelationalFunction(fun)),
                };
                write!(self.out, "{fun_name}(")?;

                self.write_expr(module, argument, func_ctx)?;

                write!(self.out, ")")?
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
        &mut self,
        module: &Module,
        handle: Handle<naga::Type>,
    ) -> BackendResult {
        let ty = &module.types[handle];
        match ty.inner {
            TypeInner::Struct { .. } => self
                .out
                .write_str(self.names[&NameKey::Type(handle)].as_str())?,
            ref other => self.write_type_inner(module, other)?,
        }

        Ok(())
    }

    fn write_type_inner(&mut self, module: &Module, inner: &TypeInner) -> BackendResult {
        match *inner {
            TypeInner::Vector { size, scalar } => write!(
                self.out,
                "{SHADER_LIB}::{}Vec{}",
                conv::upper_glam_prefix(scalar),
                conv::vector_size_str(size),
            )?,
            TypeInner::Sampler { comparison: false } => {
                write!(self.out, "{SHADER_LIB}::Sampler")?;
            }
            TypeInner::Sampler { comparison: true } => {
                write!(self.out, "{SHADER_LIB}::SamplerComparison")?;
            }
            TypeInner::Image { .. } => {
                unimplemented!()
            }
            TypeInner::Scalar(scalar) => {
                write!(self.out, "{}", unwrap_to_rust(scalar))?;
            }
            TypeInner::Atomic(scalar) => {
                write!(self.out, "atomic<{}>", unwrap_to_rust(scalar))?;
            }
            TypeInner::Array {
                base,
                size,
                stride: _,
            } => {
                write!(self.out, "[")?;
                match size {
                    naga::ArraySize::Constant(len) => {
                        self.write_type(module, base)?;
                        write!(self.out, ", {len}")?;
                    }
                    naga::ArraySize::Pending(..) => {
                        unimplemented!()
                    }
                    naga::ArraySize::Dynamic => {
                        self.write_type(module, base)?;
                    }
                }
                write!(self.out, "]")?;
            }
            TypeInner::BindingArray { .. } => {
                unimplemented!()
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => {
                if columns == rows {
                    write!(
                        self.out,
                        "{SHADER_LIB}::{}Mat{}",
                        conv::upper_glam_prefix(scalar),
                        conv::vector_size_str(columns),
                    )?;
                } else {
                    write!(
                        self.out,
                        "{SHADER_LIB}::{}Mat{}x{}",
                        conv::upper_glam_prefix(scalar),
                        conv::vector_size_str(columns),
                        conv::vector_size_str(rows),
                    )?;
                }
            }
            TypeInner::Pointer { base, space: _ } => {
                if self.flags.contains(WriterFlags::RAW_POINTERS) {
                    write!(self.out, "*mut ")?;
                } else {
                    write!(self.out, "&mut ")?;
                }
                self.write_type(module, base)?;
            }
            TypeInner::ValuePointer {
                size: _,
                scalar,
                space: _,
            } => {
                write!(self.out, "*mut {}", unwrap_to_rust(scalar))?;
            }
            TypeInner::AccelerationStructure { .. } => {
                unimplemented!()
            }
            TypeInner::Struct { .. } => {
                unreachable!("should only see a struct by name");
            }
            TypeInner::RayQuery { .. } => {
                unimplemented!()
            }
        }

        Ok(())
    }

    /// Helper method used to write global variables as translated into struct fields
    fn write_global_variable_as_field(
        &mut self,
        module: &Module,
        global: &naga::GlobalVariable,
        handle: Handle<naga::GlobalVariable>,
    ) -> BackendResult {
        // Write group and binding attributes if present
        if let Some(ref binding) = global.binding {
            self.write_attributes(&[
                Attribute::Group(binding.group),
                Attribute::Binding(binding.binding),
            ])?;
        }

        write!(
            self.out,
            " {}: ",
            &self.names[&NameKey::GlobalVariable(handle)]
        )?;

        // Write global type
        self.write_type(module, global.ty)?;

        // Write initializer
        // TODO: need to generate a separate initializer expression
        // if let Some(init) = global.init {
        //     write!(self.out, " = ")?;
        //     self.write_const_expression(module, init, &module.global_expressions)?;
        // }

        // End with comma separating from the next field
        writeln!(self.out, ",")?;

        Ok(())
    }

    /// Helper method used to write global constants
    ///
    /// # Notes
    /// Ends in a newline
    fn write_global_constant(
        &mut self,
        module: &Module,
        handle: Handle<naga::Constant>,
    ) -> BackendResult {
        let name = &self.names[&NameKey::Constant(handle)];
        // First write only constant name
        write!(self.out, "#[allow(non_upper_case_globals)]\nconst {name}: ")?;
        self.write_type(module, module.constants[handle].ty)?;
        write!(self.out, " = ")?;
        let init = module.constants[handle].init;
        self.write_const_expression(module, init, &module.global_expressions)?;
        writeln!(self.out, ";")?;

        Ok(())
    }

    fn visibility(&self) -> &'static str {
        if self.flags.contains(WriterFlags::PUBLIC) {
            "pub "
        } else {
            ""
        }
    }

    pub fn finish(self) -> W {
        self.out
    }
}
