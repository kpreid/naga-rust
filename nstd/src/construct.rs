/// Trait to adapt the Naga constructor paradigm to Rust.
pub trait New {
    type Args;
    fn new(args: Self::Args) -> Self;
}
