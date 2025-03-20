//! Tests of the exact source text produced by the Rust backend.
//!
//! TODO: Consider whether we want to rewrite some or all of these tests to use
//! `syn` or another parser to match trees instead of exact text.

use core::error::Error;
use core::fmt;

use pretty_assertions::assert_eq;

use naga_rust_back::Config;

fn translate_without_header(config: Config, wgsl_source_text: &str) -> String {
    fn inner(config: Config, wgsl_source_text: &str) -> Result<String, Box<dyn Error>> {
        let module: naga::Module = naga::front::wgsl::parse_str(wgsl_source_text)?;

        let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .subgroup_stages(naga::valid::ShaderStages::all())
        .subgroup_operations(naga::valid::SubgroupOperationSet::all())
        .validate(&module)?;

        let mut translated_source: String =
            naga_rust_back::write_string(&module, &module_info, config)?;

        // Kludge: Strip off the first boilerplate lines without caring what they are exactly.
        let header_end = translated_source
            .char_indices()
            .filter(|&(_, ch)| ch == '\n')
            .map(|(i, _)| i)
            .nth(1)
            .ok_or("header not found")?;
        translated_source.replace_range(0..=header_end, "");

        Ok(translated_source)
    }

    match inner(config, wgsl_source_text) {
        Ok(translated_source) => translated_source,
        Err(e) => panic!("{}", ErrorChain(&*e)),
    }
}

#[test]
fn visibility_control() {
    assert_eq!(
        translate_without_header(Config::new(), "fn foo() {}"),
        "#[allow(unused, clippy::all)]\nfn foo() {\n    return;\n}\n\n"
    );
    assert_eq!(
        translate_without_header(Config::new().public_items(true), "fn foo() {}"),
        "#[allow(unused, clippy::all)]\npub fn foo() {\n    return;\n}\n\n"
    );
}

#[test]
fn global_variable() {
    assert_eq!(
        translate_without_header(Config::new(), r"var<private> foo: i32 = 1;"),
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

#[test]
fn switch() {
    assert_eq!(
        translate_without_header(
            Config::new(),
            r"
            fn switching(x: i32) -> i32 {
                switch (x) {
                    case 0 { return 0; }
                    case 1, 2 { return 1; }
                    case default { return 2; }
                }
            }
            "
        ),
        indoc::indoc! {
            "
            #[allow(unused, clippy::all)]
            fn switching(x: i32) -> i32 {
                match x {
                    0i32 => {
                        return 0i32;
                    }
                    1i32 | 2i32 => {
                        return 1i32;
                    }
                    _ => {
                        return 2i32;
                    }
                }
            }

            "
        }
    );
}

#[test]
fn array_type_sizes() {
    assert_eq!(
        translate_without_header(
            Config::new(),
            r"struct Foo {
                x: array<i32, 10>,
                y: array<i32>,
            }"
        ),
        indoc::indoc! {
            "#[repr(C)]
            struct Foo {
                x: [i32; 10],
                y: [i32],
            }

            "
        }
    );
}

#[test]
fn array_length() {
    assert_eq!(
        translate_without_header(
            Config::new(),
            r"
            @group(0) @binding(1) var<storage> arr: array<u32>;
            fn length() -> u32 {
                return arrayLength(&arr);
            }
            ",
        ),
        // TODO: we don't support bindings properly so lots of this code is nonsense.
        // This test is only intending to check the translation of arrayLength(), which is
        // hard to test separately since it must take a `ptr<storage, array<..>>`.
        indoc::indoc! {
            "
            struct Globals {
                #[rt::group(0)]
                #[rt::binding(1)]
                arr: [u32],
            }
            impl Default for Globals {
                fn default() -> Self { Self {
                    arr: Default::default(),
                }}
            }
            #[allow(unused, clippy::all)]
            fn length() -> u32 {
                return (&arr).len();
            }

            "
        }
    );
}

// -------------------------------------------------------------------------------------------------

/// Formatting wrapper which prints an [`Error`] together with its `source()` chain.
///
/// We bother to do this for tests because it is way more legible than `unwrap()`'s Debug format.
/// Note that the same code exists in `naga-rust-macros` for user facing error reporting.
#[derive(Clone, Copy, Debug)]
struct ErrorChain<'a>(&'a (dyn Error + 'a));

impl fmt::Display for ErrorChain<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_error_chain(fmt, self.0)
    }
}

fn format_error_chain(fmt: &mut fmt::Formatter<'_>, mut error: &(dyn Error + '_)) -> fmt::Result {
    write!(fmt, "{error}")?;
    while let Some(source) = error.source() {
        error = source;
        write!(fmt, "\n↳ {error}")?;
    }

    Ok(())
}
