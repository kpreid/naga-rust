pub(crate) struct Gensym(pub naga::Handle<naga::Expression>);

impl core::fmt::Display for Gensym {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.write_prefixed(f, "_e")
    }
}

pub(crate) trait LevelNext {
    fn next(self) -> Self;
}
impl LevelNext for naga::back::Level {
    fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

/// Classifies address spaces into two kinds: the kind that go in our global variables struct,
/// and the kind that go in our resources struct.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlobalKind {
    Variable,
    Resource,
}

impl GlobalKind {
    /// Returns `None` if this address space can’t be a global.
    pub fn new(address_space: naga::AddressSpace) -> Option<Self> {
        match address_space {
            // Function locals — should never appear as a global.
            naga::AddressSpace::Function => None,

            // Mutable global data
            naga::AddressSpace::Private => Some(GlobalKind::Variable),
            naga::AddressSpace::WorkGroup => Some(GlobalKind::Variable),

            // Resources
            naga::AddressSpace::Uniform => Some(GlobalKind::Resource),
            naga::AddressSpace::Storage { access: _ } => Some(GlobalKind::Resource),
            naga::AddressSpace::Handle => Some(GlobalKind::Resource),

            // Unsupported
            naga::AddressSpace::Immediate => None,
            naga::AddressSpace::TaskPayload => None,
            naga::AddressSpace::RayPayload => None,
            naga::AddressSpace::IncomingRayPayload => None,
        }
    }

    pub fn of_variable(variable: &naga::GlobalVariable) -> Option<Self> {
        Self::new(variable.space)
    }

    pub fn filter(
        self,
        variables: &naga::Arena<naga::GlobalVariable>,
    ) -> impl Iterator<Item = (naga::Handle<naga::GlobalVariable>, &naga::GlobalVariable)> {
        variables
            .iter()
            .filter(move |(_, v)| Self::of_variable(v) == Some(self))
    }
}
