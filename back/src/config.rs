use alloc::borrow::Cow;
use alloc::string::String;

/// Configuration/builder for options for Rust code generation.
///
/// This configuration allows you to control syntactic characteristics of the output,
/// and also Rust features that have no equivalent in shader languages.
#[derive(Debug)]
pub struct Config {
    pub(crate) flags: WriterFlags,
    pub(crate) runtime_path: Cow<'static, str>,
    pub(crate) global_struct: Option<String>,
    #[allow(dead_code, reason = "reminding ourselves of the future")]
    pub(crate) edition: Edition,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Creates a [`Config`] with default options.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            flags: WriterFlags::empty(),
            runtime_path: Cow::Borrowed("::naga_rust_rt"),
            global_struct: None,
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
    /// The resulting code is `unsafe` and may be unsound if the input module
    /// uses pointers incorrectly.
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
    /// The default is `false`.
    #[must_use]
    pub fn public_items(mut self, value: bool) -> Self {
        self.flags.set(WriterFlags::PUBLIC, value);
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
    #[must_use]
    pub fn global_struct(mut self, name: impl Into<String>) -> Self {
        self.global_struct = Some(name.into());
        self
    }

    pub(crate) fn use_global_struct(&self) -> bool {
        self.global_struct.is_some()
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
        const PUBLIC = 0x3;
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
