@group(0) @binding(0) var<uniform> time: f32;

@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pos_int: vec2<u32> = vec2u(position.xy);

    let val: f32 = f32(pos_int.x ^ pos_int.y) * 0.01;
    let color = (vec3f(
        val - (time * 0.16f),
        val - (time * 0.12f),
        val - (time * 0.08f),
    ) % 1.0 + 1.0) % 1.0;

    // TODO: vec4(color, 1.0) should work
    return vec4(
        color.r,
        color.g,
        color.b,
        1.0,
    );
}

