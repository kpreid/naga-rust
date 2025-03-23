use naga_rust_embed::wgsl;

#[test]
pub(crate) fn switch() {
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
pub(crate) fn while_loop() {
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
