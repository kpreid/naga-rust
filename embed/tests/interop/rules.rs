use naga_rust_embed::wgsl;

// -------------------------------------------------------------------------------------------------

#[test]
fn derive_rule() {
    wgsl!(
        rule(struct(DeriveYes) => derive(::core::cmp::PartialOrd)),
        "
        struct DeriveYes { x: i32, }
        struct DeriveNo { x: i32, }
        "
    );

    // If the rule applies to too many things, this will fail to compile.
    impl PartialOrd for DeriveNo {
        fn partial_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
            unimplemented!()
        }
    }

    // If the rule applies to too few things, this will fail to compile.
    assert!(DeriveYes { x: 1 } < DeriveYes { x: 2 });
}
