//! Tests of the exact source text produced by the Rust backend.

use naga_rust_back::WriterFlags;

fn translate_without_header(flags: WriterFlags, wgsl_source_text: &str) -> String {
    let module: naga::Module = naga::front::wgsl::parse_str(wgsl_source_text).unwrap();

    let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module)
    .unwrap();

    let mut translated_source: String =
        naga_rust_back::write_string(&module, &module_info, flags).unwrap();

    // Kludge: Strip off the first boilerplate lines without caring what they are exactly.
    let header_end = translated_source
        .char_indices()
        .filter(|&(_, ch)| ch == '\n')
        .map(|(i, _)| i)
        .nth(2)
        .unwrap();
    translated_source.replace_range(0..header_end + 1, "");

    translated_source
}

#[test]
fn visibility_control() {
    assert_eq!(
        translate_without_header(WriterFlags::empty(), "fn foo() {}"),
        "fn foo() {\n    return;\n}\n\n"
    );
    assert_eq!(
        translate_without_header(WriterFlags::PUBLIC, "fn foo() {}"),
        "pub fn foo() {\n    return;\n}\n\n"
    );
}
