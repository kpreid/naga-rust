//! Tests of the exact source text produced by the Rust backend.

#![allow(clippy::needless_pass_by_value)]

use core::error::Error as ErrorTrait;
use core::fmt;

use naga_rust_back::Config;

// -------------------------------------------------------------------------------------------------

mod control_flow;
mod functions;
mod globals;
mod misc;
mod structs;

// -------------------------------------------------------------------------------------------------

/// Run the Naga WGSL frontend in preparation for translation.
fn frontend(
    wgsl_source_text: &str,
) -> Result<(naga::Module, naga::valid::ModuleInfo), Box<dyn ErrorTrait>> {
    let module: naga::Module = naga::front::wgsl::parse_str(wgsl_source_text)?;

    let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module)?;

    Ok((module, module_info))
}

/// Translate a module, panicking on error.
fn translate(config: Config, wgsl_source_text: &str) -> String {
    fn inner(config: Config, wgsl_source_text: &str) -> Result<String, Box<dyn ErrorTrait>> {
        let (module, module_info) = frontend(wgsl_source_text)?;
        Ok(naga_rust_back::write_string(&module, &module_info, config)?)
    }

    match inner(config, wgsl_source_text) {
        Ok(translated_source) => translated_source,
        Err(e) => panic!("{}", ErrorChain(&*e)),
    }
}

/// Translate a function body, panicking on error or if there is more than one function.
fn translate_body(config: Config, wgsl_source_text: &str) -> String {
    fn inner(config: Config, wgsl_source_text: &str) -> Result<String, Box<dyn ErrorTrait>> {
        let (module, module_info) = frontend(wgsl_source_text)?;
        Ok(naga_rust_back::translate_function_body_only_for_testing(
            &module,
            &module_info,
            &config,
        )?)
    }

    match inner(config, wgsl_source_text) {
        Ok(translated_source) => translated_source,
        Err(e) => panic!("{}", ErrorChain(&*e)),
    }
}

#[track_caller]
fn expect_error(config: Config, wgsl_source_text: &str) -> naga_rust_back::Error {
    let module: naga::Module = naga::front::wgsl::parse_str(wgsl_source_text).unwrap();
    let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module)
    .unwrap();

    match naga_rust_back::write_string(&module, &module_info, config) {
        Ok(_) => panic!("expected error, but got success"),
        Err(e) => e,
    }
}

// -------------------------------------------------------------------------------------------------

/// Formatting wrapper which prints an [`Error`] together with its `source()` chain.
///
/// We bother to do this for tests because it is way more legible than `unwrap()`'s Debug format.
/// Note that the same code exists in `naga-rust-macros` for user facing error reporting.
#[derive(Clone, Copy, Debug)]
struct ErrorChain<'a>(&'a (dyn ErrorTrait + 'a));

impl fmt::Display for ErrorChain<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_error_chain(fmt, self.0)
    }
}

fn format_error_chain(
    fmt: &mut fmt::Formatter<'_>,
    mut error: &(dyn ErrorTrait + '_),
) -> fmt::Result {
    write!(fmt, "{error}")?;
    while let Some(source) = error.source() {
        error = source;
        write!(fmt, "\n↳ {error}")?;
    }

    Ok(())
}
