/// Configuration/builder for options for Rust code generation.
///
/// This configuration allows you to control syntactic characteristics of the output,
/// and also Rust features that have no equivalent in shader languages.
#[derive(Debug)]
pub struct Config {
    pub(crate) flags: WriterFlags,
}

impl Default for Config {
    #[must_use]
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
