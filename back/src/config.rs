use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::logic::{GlobalLocation, GlobalTranslation};
use crate::ra;

// -------------------------------------------------------------------------------------------------

/// Configuration/builder for options for Rust code generation.
///
/// This configuration allows you to control syntactic characteristics of the output,
/// and also Rust features that have no equivalent in shader languages.
#[derive(Clone, Debug)]
pub struct Config {
    pub(crate) flags: WriterFlags,
    pub(crate) runtime_path: Cow<'static, str>,
    pub(crate) global_struct: Option<String>,
    pub(crate) resource_struct: Option<String>,
    #[allow(dead_code, reason = "reminding ourselves of the future")]
    pub(crate) edition: Edition,
    pub(crate) rules: Vec<Rule>,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    // When adding new options, also add them to:
    // * `ConfigAndStr::parse` in `macros/src/parse_config.rs`.
    // * `embed/src/configuration_syntax.md` (documentation for the macros).

    /// Creates a [`Config`] with default options.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            flags: WriterFlags::INCLUDE_FUNCTIONS,
            runtime_path: Cow::Borrowed("::naga_rust_rt"),
            global_struct: None,
            resource_struct: None,
            edition: Edition::Rust2024,
            rules: Vec::new(),
        }
    }

    /// Sets whether to translate functions, rather than ignoring them.
    ///
    /// This may be disabled to produce an output containing only `struct`s and `const`s,
    /// and remove any requirement to specify a [`global_struct`][Self::global_struct] or
    /// [`resource_struct`][Self::resource_struct].
    ///
    /// The default is `true`.
    #[must_use]
    pub fn include_functions(mut self, value: bool) -> Self {
        self.flags.set(WriterFlags::INCLUDE_FUNCTIONS, value);
        self
    }

    /// Sets whether the generated code contains explicit types when they could be omitted.
    ///
    /// The default is `false`.
    #[must_use]
    pub fn explicit_types(mut self, value: bool) -> Self {
        self.flags.set(WriterFlags::EXPLICIT_TYPES, value);
        self
    }

    /// Sets whether the generated code uses raw pointers instead of references.
    ///
    /// The resulting code may be unsound if the input module uses pointers incorrectly.
    ///
    /// <div class="warning">
    ///
    /// Currently, this does not actually work, in that it generates code which does not
    /// even try to `unsafe`ly dereference the raw pointers it uses.
    /// The exact behavior of this option is not yet decided, but it will likely cause the
    /// generated functions to be `unsafe fn`s.
    ///
    /// </div>
    ///
    /// The default is `false`.
    ///
    /// TODO: This should be configurable on a per-function basis.
    #[must_use]
    pub fn raw_pointers(mut self, value: bool) -> Self {
        self.flags.set(WriterFlags::RAW_POINTERS, value);
        self
    }

    /// Sets whether generated items have `pub` visibility instead of private.
    ///
    /// This option applies to all functions, methods, constants, and structs, and
    /// all fields of generated structs.
    /// It affects both structs translated from the shader code, and the
    /// [`global_struct`][Self::global_struct] and [`resource_struct`][Self::resource_struct]
    /// if present.
    ///
    /// The default is `false`.
    #[must_use]
    pub fn public_items(mut self, value: bool) -> Self {
        self.flags.set(WriterFlags::PUBLIC, value);
        self
    }

    /// Sets whether to allow the generated code to panic on entering code that cannot be
    /// translated, rather than failing generation.
    ///
    /// This applies to all unsupported expressions and statements, but not to unsupported types.
    ///
    /// The default is `false`.
    #[must_use]
    pub fn allow_unimplemented(mut self, value: bool) -> Self {
        self.flags.set(WriterFlags::ALLOW_UNIMPLEMENTED, value);
        self
    }

    /// Sets the Rust module path to the runtime support library.
    ///
    /// The default is `"::naga_rust_rt"`.
    ///
    /// # Panics
    ///
    /// May panic if the path is not syntactically valid or not an absolute path.
    #[must_use]
    pub fn runtime_path(mut self, value: impl Into<Cow<'static, str>>) -> Self {
        let value = value.into();
        assert!(
            value.starts_with("::") || value.starts_with("crate::"),
            "path should be an absolute path"
        );
        self.runtime_path = value;
        self
    }

    /// Allow declarations of global variables, generate a struct with the given `name` to hold
    /// them, and make all functions methods of that struct.
    ///
    /// The struct has one constructor method, which is declared as either
    /// `const fn new()` or `const fn new(resources: &ResourceStructName)`
    /// depending on whether [`resource_struct()`][Self::resource_struct] is also set.
    /// If there are no parameters, then it also implements [`Default`].
    ///
    /// If this option is not set, shaders may not contain declarations of variables with
    /// [address spaces] `private` or `workgroup`.
    ///
    /// [address spaces]: https://www.w3.org/TR/WGSL/#address-space
    #[must_use]
    pub fn global_struct(mut self, name: impl Into<String>) -> Self {
        self.global_struct = Some(name.into());
        self
    }

    /// Allow declarations of resources (e.g. uniforms), generate a struct with the given `name` to
    /// hold them, and, if [`global_struct()`][Self::global_struct] is not also set,
    /// make all functions methods of that struct.
    ///
    /// If this option is not set, shaders may not contain declarations of variables with
    /// [address spaces] `uniform` or `storage`.
    ///
    /// [address spaces]: https://www.w3.org/TR/WGSL/#address-space
    #[must_use]
    pub fn resource_struct(mut self, name: impl Into<String>) -> Self {
        self.resource_struct = Some(name.into());
        self
    }

    /// Adds a rule.
    ///
    /// Rules modify the translation of specific parts of the shader code.
    /// When multiple rules apply and their effects conflict, rules added later take priority over
    /// rules added earlier.
    #[must_use]
    pub fn rule(mut self, rule: impl Into<Rule>) -> Self {
        self.rules.push(rule.into());
        self
    }
}

/// Internal methods that help generate code based on this config.
impl Config {
    /// Returns whether we should generate functions instead of free functions.
    pub(crate) fn functions_are_methods(&self) -> bool {
        self.global_struct.is_some() || self.resource_struct.is_some()
    }

    /// Returns what the self type of our `impl` block is, if we have one.
    pub(crate) fn impl_type(&self) -> Option<&str> {
        match self.global_struct {
            Some(ref name) => Some(name),
            None => self.resource_struct.as_deref(),
        }
    }

    /// Returns the expression for the struct whose fields are the translation of
    /// shader global variables.
    pub(crate) fn global_field_access_expr(
        &self,
        module: &naga::Module,
        variable: &naga::GlobalVariable,
    ) -> ra::Expr {
        match (
            GlobalTranslation::get(module, variable).location,
            &self.global_struct,
        ) {
            // If we have both resource struct and global struct, the resource struct is
            // nested inside the global struct.
            (GlobalLocation::Resource, Some(_)) => {
                ra::Expr::NamedField(Box::new(ra::Expr::Self_), "resources".into())
            }
            (GlobalLocation::Resource, None) | (GlobalLocation::Variable, _) => ra::Expr::Self_,
        }
    }

    /// Iterates over the effects of all rules that apply in the given situation.
    ///
    /// Later rules should take priority over earlier rules (e.g. by overwriting a variable).
    pub(crate) fn apply_rules(&self, input: &RuleInput<'_>) -> impl Iterator<Item = &Effect> {
        self.rules.iter().flat_map(
            |Rule {
                 conditions: condition,
                 effects,
             }| {
                if condition.iter().all(|c| c.test(input)) {
                    effects.as_slice()
                } else {
                    &[]
                }
            },
        )
    }
}

bitflags::bitflags! {
    /// Options for what Rust code is generated.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) struct WriterFlags: u32 {
        /// Include translated function definitions rather than omitting them.
        ///
        /// If this is not set, only `struct` and `const` items are produced.
        const INCLUDE_FUNCTIONS = 1 << 0;

        /// Always annotate the type information instead of inferring.
        const EXPLICIT_TYPES = 1 << 1;

        /// Generate code using raw pointers instead of references.
        /// The resulting code is `unsafe` and may be unsound if the input module
        /// uses pointers incorrectly.
        const RAW_POINTERS = 1 << 2;

        /// Generate items with `pub` visibility instead of private.
        const PUBLIC = 1 << 3;

        /// Allow the generated code to panic on entering code that cannot be
        /// translated, rather than failing generation.
        const ALLOW_UNIMPLEMENTED = 1 << 4;
    }
}

/// Edition of Rust code to generate.
///
/// We currently only support one edition, but this exists anyway to prepare to document
/// any edition dependencies in the code generator.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Edition {
    Rust2024,
}

// -------------------------------------------------------------------------------------------------

/// Controls some aspect of the translation from shader code to Rust that needs Rust-specific
/// choices that cannot be expressed inside the shader code.
///
/// Put these in [`Config::rule()`].
///
/// When multiple rules apply and their effects conflict, rules added later take priority over
/// rules added earlier.
#[derive(Clone, Debug, Eq, PartialEq)]
#[allow(clippy::exhaustive_structs)]
pub struct Rule {
    /// The rule applies when all of these conditions are met.
    pub conditions: Vec<Condition>,
    /// The rule has all of these effects.
    pub effects: Vec<Effect>,
}

/// Specifies a condition under which a [`Rule`] applies to a part of the input shader code.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Condition {
    /// Applies to the shader function with the given name.
    Function(String),
    /// Applies to the struct with the given name (excluding resource and global structs).
    Struct(String),
    // When adding a variant, update the parser and documentation in `naga-rust-embed` too.
}

/// Specifies the effect of a [`Rule`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Effect {
    /// Applies a derive macro to translated structs.
    ///
    /// The given string is copied literally into the output and must be a path to a derive macro,
    /// as accepted by the `#[derive]` attribute.
    Derive(String),

    /// Applies the `#[inline]` attribute to translated functions.
    ///
    /// If multiple `Inline` effects apply, then for each function,
    /// the last choice of inlining wins.
    ///
    /// See [The Rust Reference](https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute)
    /// for information on how the `#[inline]` attribute is interpreted.
    ///
    /// Caution: Using this attribute is not always necessary to obtain inlining,
    /// and inlining is not always beneficial. Usage should be guided by profiling and benchmarks.
    Inline(Inline),
    // When adding a variant, update the parser and documentation in `naga-rust-embed` too.
}

/// What `#[inline]` attribute [`Effect::Inline`] produces.
///
/// See [The Rust Reference](https://doc.rust-lang.org/reference/attributes/codegen.html#the-inline-attribute)
/// for information on how the `#[inline]` attribute is interpreted.
///
/// Caution: Using this attribute is not always necessary to obtain inlining,
/// and inlining is not always beneficial. Usage should be guided by profiling and benchmarks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Inline {
    /// `#[inline]`
    Maybe,
    /// `#[inline(always)]`
    Always,
    /// `#[inline(never)]`
    Never,
}

/// Data consulted by rules.
pub(crate) struct RuleInput<'a> {
    /// Shader function name.
    /// `None` if rules are being applied to something other than a function item,
    /// or a generated function not corresponding to a shader function.
    pub function: Option<&'a str>,

    /// Shader struct name.
    /// `None` if rules are being applied to something other than a struct item,
    /// or a generated struct not corresponding to a shader struct.
    pub r#struct: Option<&'a str>,
}

impl From<(Condition, Effect)> for Rule {
    fn from((c, e): (Condition, Effect)) -> Self {
        Self {
            conditions: vec![c],
            effects: vec![e],
        }
    }
}

impl From<Effect> for Rule {
    /// Creates a rule that always applies.
    fn from(e: Effect) -> Self {
        Self {
            conditions: vec![],
            effects: vec![e],
        }
    }
}

impl Condition {
    fn test(&self, input: &RuleInput<'_>) -> bool {
        match *self {
            Condition::Function(ref name) => input.function == Some(name.as_str()),
            Condition::Struct(ref name) => input.r#struct == Some(name.as_str()),
        }
    }
}
