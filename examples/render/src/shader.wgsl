@group(0) @binding(0) var<uniform> time: f32;

@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return xor_pattern(position);
}

fn xor_pattern(position: vec4f) -> vec4<f32> {
    let pos_int: vec2<u32> = vec2u(position.xy);

    const time_scales: vec3f = vec3f(0.16, -0.151, 0.03);
    
    let val: f32 = f32(pos_int.x ^ pos_int.y) * 0.01;
    let color = ((vec3f(val) * 0.125 - (time_scales * time)) % 1.0 + 1.0) % 1.0;

    return vec4(color, 1.0);
}
