fn add_one(x: i32) -> i32 {
    return x + 1;
}

struct StructTest {
    a: i32,
    b: f32,
}

fn modify_struct(s_ptr: ptr<function, StructTest>) {
    (*s_ptr).a += 1;
    (*s_ptr).b += 1.0;
}
