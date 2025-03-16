mod shade {
    include!(concat!(env!("OUT_DIR"), "/translated.rs"));
}

#[test]
fn test_add_one() {
    assert_eq!(shade::add_one(10), 11);
}

#[test]
fn use_struct_definition() {
    let mut s = shade::StructTest { a: 1, b: 2.0 };
    shade::modify_struct(&mut s);
}
