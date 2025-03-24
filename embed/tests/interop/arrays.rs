use naga_rust_embed::wgsl;
use naga_rust_rt::Scalar;

#[test]
pub(crate) fn array_ctor() {
    wgsl!(
        r"fn f() -> i32 {
            var a: array<i32, 4> = array(10i, 20, 30, 40);
            return a[2];
        }"
    );
    assert_eq!(f(), 30);
}

#[test]
pub(crate) fn array_access_fixed() {
    wgsl!(
        r"
        fn modify_array(a_ptr: ptr<private, array<u32, 2>>) {
            (*a_ptr)[0] += 1;
            (*a_ptr)[1] += 2;
        }
        "
    );

    // TODO: The type expected ought to be [u32; 2] instead, but it will require further work
    // to make that actually true.
    let mut a = [Scalar(10), Scalar(100)];
    modify_array(&mut a);
    assert_eq!(a, [Scalar(11), Scalar(102)]);
}
