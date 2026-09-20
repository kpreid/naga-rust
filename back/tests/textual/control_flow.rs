use pretty_assertions::assert_eq;

use naga_rust_back::Config;

use crate::translate_body;

// -------------------------------------------------------------------------------------------------

#[test]
fn switch() {
    // TODO: we can’t fully exercise `fall_through` without using an input syntax other than WGSL
    assert_eq!(
        translate_body(
            Config::new(),
            r"fn switching(x: i32) -> i32 {
                switch (x) {
                    case 0 { return 0; }
                    case 1, 2 { return 1; }
                    case default { return 2; }
                }
            }"
        ),
        indoc::indoc! { "{
            match ::naga_rust_rt::Scalar::into_inner(x) {
                0i32 => {
                    return ::naga_rust_rt::Scalar(0i32);
                }
                1i32 | 2i32 => {
                    return ::naga_rust_rt::Scalar(1i32);
                }
                _ => {
                    return ::naga_rust_rt::Scalar(2i32);
                }
            }
        }" }
    );
}

#[test]
fn continuing() {
    assert_eq!(
        translate_body(
            Config::new(),
            r"fn foo() {
                var i = 0;
                loop {
                    i += 1;
                    continuing { i += 1; }
                }
            }",
        ),
        indoc::indoc! {"{
            let mut i: ::naga_rust_rt::Scalar<i32> = ::naga_rust_rt::Scalar(0i32);
        
            'naga_break: loop {
                'naga_continue: {
                    let _e2 = i;
                    i = (_e2 + ::naga_rust_rt::Scalar(1i32));
                }
                let _e5 = i;
                i = (_e5 + ::naga_rust_rt::Scalar(1i32));
            }
            return;
        }"}
    );
}

#[test]
fn continuing_break_if() {
    assert_eq!(
        translate_body(
            Config::new(),
            r"fn foo() {
                var i = 0;
                loop {
                    i += 1;
                    continuing { 
                        i += 1;
                        break if i >= 10;
                    }
                }
            }",
        ),
        indoc::indoc! {"{
            let mut i: ::naga_rust_rt::Scalar<i32> = ::naga_rust_rt::Scalar(0i32);
        
            'naga_break: loop {
                'naga_continue: {
                    let _e2 = i;
                    i = (_e2 + ::naga_rust_rt::Scalar(1i32));
                }
                let _e5 = i;
                i = (_e5 + ::naga_rust_rt::Scalar(1i32));
                let _e8 = i;
                if ::naga_rust_rt::Scalar::into_branch_condition(_e8.elementwise_ge(::naga_rust_rt::Scalar(10i32))) {
                    break 'naga_break;
                }
            }
            return;
        }"}
    );
}

#[test]
fn if_without_else() {
    assert_eq!(
        translate_body(
            Config::new(),
            r"fn foo(x: i32) -> i32 {
                if x > 0 {
                    return 1;
                }
                return -1;
            }",
        ),
        indoc::indoc! {"{
            if ::naga_rust_rt::Scalar::into_branch_condition(x.elementwise_gt(::naga_rust_rt::Scalar(0i32))) {
                return ::naga_rust_rt::Scalar(1i32);
            }
            return ::naga_rust_rt::Scalar(-1i32);
        }"}
    );
}

#[test]
fn if_else() {
    assert_eq!(
        translate_body(
            Config::new(),
            r"fn foo(x: i32) -> i32 {
                if x > 0 {
                    return 1;
                } else {
                    return -1;
                }
            }",
        ),
        indoc::indoc! {"{
            if ::naga_rust_rt::Scalar::into_branch_condition(x.elementwise_gt(::naga_rust_rt::Scalar(0i32))) {
                return ::naga_rust_rt::Scalar(1i32);
            } else {
                return ::naga_rust_rt::Scalar(-1i32);
            }
        }"}
    );
}

#[test]
fn if_else_chain() {
    assert_eq!(
        translate_body(
            Config::new(),
            r"fn signum(x: i32) -> i32 {
                if x > 0 {
                    return 1;
                } else if x < 0 {
                    return -1;
                } else {
                    return 0;
                }
            }",
        ),
        indoc::indoc! {"{
            if ::naga_rust_rt::Scalar::into_branch_condition(x.elementwise_gt(::naga_rust_rt::Scalar(0i32))) {
                return ::naga_rust_rt::Scalar(1i32);
            } else {
                if ::naga_rust_rt::Scalar::into_branch_condition(x.elementwise_lt(::naga_rust_rt::Scalar(0i32))) {
                    return ::naga_rust_rt::Scalar(-1i32);
                } else {
                    return ::naga_rust_rt::Scalar(0i32);
                }
            }
        }"}
    );
}

#[test]
fn discard() {
    assert_eq!(
        translate_body(
            Config::new(),
            r"fn foo() {
                discard;
            }",
        ),
        indoc::indoc! {"{
            ::naga_rust_rt::discard();
        }"}
    );
}
