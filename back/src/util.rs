pub(crate) struct Baked(pub naga::Handle<naga::Expression>);

impl core::fmt::Display for Baked {
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
