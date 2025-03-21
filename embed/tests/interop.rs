//! Tests which actually execute shader code and exercise interoperation with ordinary Rust.

use exhaust::Exhaust as _;

// TODO: Should there should be an explicit public vector-type API module which is not rt::?
use naga_rust_embed::rt::Vec2;
use naga_rust_embed::wgsl;

#[test]
fn call_entry_point() {
    wgsl!(
        r"@fragment
        fn frag(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
            return position + 1.0;
        }"
    );
    assert_eq!(
        frag(rt::Vec4::new(1.0, 2.0, 0.0, 0.0)),
        rt::Vec4::new(2.0, 3.0, 1.0, 1.0)
    );
}

#[test]
fn global_constant() {
    wgsl!("const X: f32 = 1234.0;");
    assert_eq!(X, 1234.0);
}

#[test]
fn local_constant() {
    wgsl!(
        r"fn foo() -> f32 {
            const X: f32 = 1234.0;
            return X;
        }"
    );
    assert_eq!(foo(), 1234.0);
}

#[test]
fn scalar_arithmetic() {
    wgsl!(
        r"fn add_one(x: i32) -> i32 {
            return x + 1;
        }"
    );

    assert_eq!(add_one(10), 11);
}
#[test]
fn vector_arithmetic() {
    wgsl!(
        r"fn add_one(x: vec2f) -> vec2f {
            return x + vec2f(1.0);
        }"
    );

    assert_eq!(add_one(Vec2::new(0.5, 10.0)), Vec2::new(1.5, 11.0));
}

#[test]
fn vector_mixed_construction() {
    wgsl!(
        r"fn foo() -> vec4f {
            return vec4f(1.0, vec2f(2.0, 3.0), 4.0);
        }"
    );

    assert_eq!(foo(), rt::Vec4::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn vector_cast() {
    wgsl!(
        r"fn func(x: vec2f) -> vec2i {
            return vec2i(x);
        }"
    );

    // Expect truncation
    assert_eq!(func(Vec2::new(1.5, -1.5)), Vec2::new(1, -1));
}

#[test]
fn scalar_pointer() {
    wgsl!(
        r"fn add_one_ptr(p: ptr<function, i32>) {
            *p += 1;
        }"
    );

    let mut x = 10;
    add_one_ptr(&mut x);
    assert_eq!(x, 11);
}

#[test]
fn vector_pointer() {
    wgsl!(
        r"fn add_to_ptr(p: ptr<function, vec2i>) {
            (*p).x += 1;
            (*p).y += 2;
        }"
    );

    let mut x = Vec2::new(10, 10);
    add_to_ptr(&mut x);
    assert_eq!(x, Vec2::new(11, 12));
}

#[test]
fn declare_and_modify_struct() {
    wgsl!(
        r"
        struct StructTest {
            a: i32,
            b: f32,
        }

        fn modify_struct(s_ptr: ptr<function, StructTest>) {
            (*s_ptr).a += 1;
            (*s_ptr).b += 1.0;
        }
        "
    );

    let mut s = StructTest { a: 1, b: 2.0 };
    modify_struct(&mut s);
    assert!(matches!(s, StructTest { a: 2, b: 3.0 }));
}

#[test]
fn logical_ops() {
    wgsl!(
        r"fn logic(a: bool, b: bool, c: bool) -> bool {
            return a && b || c;
        }"
    );

    for [a, b, c] in <[bool; 3]>::exhaust() {
        assert_eq!(logic(a, b, c), a && b || c);
    }
}

// TODO: implement atomics
// #[test]
// fn atomics() {
//     wgsl!(
//         global_struct = Globals,
//         r"
//         @group(0) @binding(0)
//         var<storage, read_write> storage_atomic_scalar: atomic<u32>;
//
//         fn atomic_ops() {
//             atomicAdd(&storage_atomic_scalar, 10u);
//             atomicAnd(&storage_atomic_scalar, 3u);
//         }
//         "
//     );
//
//     let globals = Globals {
//         x: core::sync::atomic::AtomicU32::new(2);
//     };
//     globals.atomic_ops();
//     assert_eq!(globals.x.load(core::sync::atomic::Ordering::Relaxed), 36);
// }

#[test]
fn array_access_fixed() {
    wgsl!(
        r"
        fn modify_array(a_ptr: ptr<private, array<u32, 2>>) {
            (*a_ptr)[0] += 1;
            (*a_ptr)[1] += 2;
        }
        "
    );

    let mut a = [10, 100];
    modify_array(&mut a);
    assert_eq!(a, [11, 102]);
}

#[test]
fn switch() {
    wgsl!(
        r"fn switching(x: i32) -> i32 {
            switch (x) {
                case 0 { return 0; }
                case 1 { return 1; }
                case default { return 2; }
            }
        }"
    );

    assert_eq!(switching(0), 0);
    assert_eq!(switching(1), 1);
    assert_eq!(switching(2), 2);
}

#[test]
fn while_loop() {
    wgsl!(
        r"fn count(n: i32) -> i32 {
            var i: i32 = 0;
            while i < n {
                i += 1;
            }
            return i;
        }"
    );

    assert_eq!(count(10), 10);
}

#[test]
fn function_call() {
    wgsl!(
        r"
        fn func0() -> i32 {
            return 10;
        }
        fn func1(x: i32) -> i32 {
            return x * 3;
        }
        fn func2(x: i32, y: i32) -> i32 {
            return x + y;
        }
        fn funcs() -> i32 {
            return func0() + func1(2) + func2(1000, 6000);
        }
        "
    );

    assert_eq!(funcs(), 7016);
}

#[test]
fn global_uniform_binding() {
    wgsl!(
        global_struct = Globals,
        r"
        struct Uniforms {
            @location(0) x: f32,
            @location(1) y: f32,
        };

        @group(0) @binding(0) var<uniform> ub: Uniforms;

        fn main() -> vec2<f32> {
            return vec2(ub.x, ub.y);
        }
    "
    );

    // TODO: impl Default is not the proper path for this -- codegen is unconditionally using
    // Default for all globals. There should instead be a constructor for the Globals struct
    // (...or, actually, a version of it which contains only bindings and not workgroup or private
    // variables).
    impl Default for Uniforms {
        fn default() -> Self {
            Uniforms { x: 1.0, y: 2.0 }
        }
    }

    assert_eq!(Globals::default().main(), Vec2::new(1.0, 2.0))
}
