use naga_rust_embed::wgsl;

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
fn loop_simple() {
    wgsl!(
        r"fn count(n: i32) -> i32 {
            var i: i32 = 0;
            loop {
                if i >= n {
                    break;
                }
                i += 1;
            }
            return i;
        }"
    );

    assert_eq!(count(10), 10);
}

#[test]
fn loop_continuing() {
    wgsl!(
        r"fn count_except_five(n: i32) -> i32 {
            var i: i32 = 0;
            var count: i32 = 0;
            loop {
                if i >= n {
                    break;
                }
                if i == 5 {
                    continue;
                }
                count += 1;
                continuing {
                    i += 1;
                }
            }
            return count;
        }"
    );

    assert_eq!(count_except_five(10), 9);
}

#[test]
fn loop_continuing_break_if() {
    wgsl!(
        r"fn count_except_five_and_at_least_one(n: i32) -> i32 {
            var i: i32 = 0;
            var count: i32 = 0;
            loop {
                if i == 5 {
                    continue;
                }
                count += 1;
                continuing {
                    i += 1;
                    break if i >= n;
                }
            }
            return count;
        }"
    );

    assert_eq!(count_except_five_and_at_least_one(0), 1);
    assert_eq!(count_except_five_and_at_least_one(10), 9);
}
