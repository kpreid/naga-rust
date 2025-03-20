#![no_main]

use libfuzzer_sys::fuzz_target;

use pretty_assertions::assert_eq;

use naga::{Module, valid::ModuleInfo};

// -------------------------------------------------------------------------------------------------

fuzz_target!(|module: ValidModule| try_build(module));

fn try_build(ValidModule(module, module_info): ValidModule) {
    let config = naga_rust_back::Config::default();

    let mut translated_source: String =
        naga_rust_back::write_string(&module, &module_info, config).unwrap();

    // TODO: try to build the source...or at least parse it
}

// -------------------------------------------------------------------------------------------------

#[derive(Debug)]
struct ValidModule(Module, ModuleInfo);

impl<'a> arbitrary::Arbitrary<'a> for ValidModule {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let module = Module::arbitrary(u)?;

        let module_info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga_rust_back::CAPABILITIES,
        )
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&module)
        .map_err(|_| arbitrary::Error::IncorrectFormat)?;

        Ok(ValidModule(module, module_info))
    }
}
