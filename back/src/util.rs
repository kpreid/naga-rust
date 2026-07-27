pub(crate) struct Gensym(pub naga::Handle<naga::Expression>);

impl core::fmt::Display for Gensym {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.write_prefixed(f, "_e")
    }
}
