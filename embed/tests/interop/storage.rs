use naga_rust_embed::{rt, wgsl};

#[test]
fn storage_read() {
    wgsl!(
        resource_struct = Resources,
        r"
        @group(0) @binding(1) var<storage, read> arr: array<u32>;
        
        fn get_the_length() -> u32 {
            return arrayLength(&arr);
        }
        "
    );

    let resources = Resources {
        // TODO: Type translation is incorrect and shouldn’t be expecting rt::Scalar
        arr: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(rt::Scalar),
    };
    assert_eq!(resources.get_the_length(), 10);

    // TODO: also read elements
}
