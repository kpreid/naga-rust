//! Tests of the exact source text produced by the Rust backend.

use pretty_assertions::assert_eq;

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
        .nth(1)
        .unwrap();
    translated_source.replace_range(0..=header_end, "");

    translated_source
}

#[test]
fn visibility_control() {
    assert_eq!(
        translate_without_header(WriterFlags::empty(), "fn foo() {}"),
        "#[allow(unused, clippy::all)]\nfn foo() {\n    return;\n}\n\n"
    );
    assert_eq!(
        translate_without_header(WriterFlags::PUBLIC, "fn foo() {}"),
        "#[allow(unused, clippy::all)]\npub fn foo() {\n    return;\n}\n\n"
    );
}

#[test]
fn global_variable() {
    assert_eq!(
        translate_without_header(WriterFlags::empty(), r"var<private> foo: i32 = 1;"),
        indoc::indoc! {
            "
            struct Globals {
                foo: i32,
            }
            impl Default for Globals {
                fn default() -> Self { Self {
                    foo: 1i32,
                }}
            }
            "
        }
    );
}
