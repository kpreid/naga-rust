use alloc::borrow::Cow;
use alloc::string::String;

use crate::util::GlobalKind;

/// Configuration/builder for options for Rust code generation.
///
/// This configuration allows you to control syntactic characteristics of the output,
/// and also Rust features that have no equivalent in shader languages.
#[derive(Debug)]
pub struct Config {
    pub(crate) flags: WriterFlags,
    pub(crate) runtime_path: Cow<'static, str>,
    pub(crate) global_struct: Option<String>,
    pub(crate) resource_struct: Option<String>,
    #[allow(dead_code, reason = "reminding ourselves of the future")]
    pub(crate) edition: Edition,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    // When adding new options, also add them to:
    // * `ConfigAndStr::parse` in `macros/src/lib.rs`.
    // * `embed/src/configuration_syntax.md` (documentation for the macros).

    /// Creates a [`Config`] with default options.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            flags: WriterFlags::empty(),
            runtime_path: Cow::Borrowed("::naga_rust_rt"),
            global_struct: None,
            resource_struct: None,
            edition: Edition::Rust2024,
        }
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
    /// This option applies to all functions or methods, and all fields of generated structs.
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

    pub(crate) fn global_field_access_prefix(
        &self,
        variable: &naga::GlobalVariable,
    ) -> &'static str {
        match (GlobalKind::of_variable(variable), &self.global_struct) {
            // If we have both resource struct and global struct, the resource struct is
            // nested inside the global struct.
            (Some(GlobalKind::Resource), Some(_)) => "self.resources.",
            (Some(GlobalKind::Resource), None) | (Some(GlobalKind::Variable), _) => "self.",
            _ => unreachable!(),
        }
    }
}

bitflags::bitflags! {
    /// Options for what Rust code is generated.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) struct WriterFlags: u32 {
        /// Always annotate the type information instead of inferring.
        const EXPLICIT_TYPES = 0x1;

        /// Generate code using raw pointers instead of references.
        /// The resulting code is `unsafe` and may be unsound if the input module
        /// uses pointers incorrectly.
        const RAW_POINTERS = 0x2;

        /// Generate items with `pub` visibility instead of private.
        const PUBLIC = 0x4;

        /// Allow the generated code to panic on entering code that cannot be
        /// translated, rather than failing generation.
        const ALLOW_UNIMPLEMENTED = 0x8;
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
