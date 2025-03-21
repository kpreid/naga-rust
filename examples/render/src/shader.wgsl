@group(0) @binding(0) var<uniform> time: f32;

@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pos_int: vec2<u32> = vec2u(position.xy);

    const time_scales: vec3f = vec3f(0.16, 0.12, 0.08);
    
    let val: f32 = f32(pos_int.x ^ pos_int.y) * 0.01;
    let color = ((vec3f(val) - (time_scales * time)) % 1.0 + 1.0) % 1.0;

    return vec4(color, 1.0);
}

