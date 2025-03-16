use std::path::PathBuf;
use std::{env, fs};

use naga_rust::naga;

fn main() {
    let module = naga::front::wgsl::parse_str(include_str!("tests/input.wgsl")).unwrap();

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module)
    .unwrap();

    let flags = naga_rust::WriterFlags::empty();

    let rust_string = naga_rust::write_string(&module, &info, flags).expect("WGSL write failed");
    let output_path = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("translated.rs");
    fs::write(output_path, rust_string).unwrap();

    // signal that our only dependencies are our own source (including include_str)
    println!("cargo::rerun-if-changed=build.rs");
}
