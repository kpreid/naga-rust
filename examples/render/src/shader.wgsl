@group(0) @binding(0) var<uniform> time: f32;

@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4(mix(vec3(0.0), xor_pattern(position), ripples(position)), 1.0);
}

fn xor_pattern(position: vec4f) -> vec3<f32> {
    let pos_int: vec2<u32> = vec2u(position.xy);

    const time_scales: vec3f = vec3f(0.16, -0.151, 0.03);
    
    let val: f32 = f32(pos_int.x ^ pos_int.y) * 0.01;
    
    return ((vec3f(val) * 0.125 - (time_scales * time)) % 1.0 + 1.0) % 1.0;
}

fn ripples(position: vec4f) -> f32 {
    let ripples =
        ripple(position.xy, vec2f(1.0, 0.0), 3.0) * 0.3
        + ripple(position.xy, vec2f(0.0, 1.0), 2.0) * 0.8
        + ripple(position.xy, vec2f(-0.7, -0.7), 3.0) * 0.4
        + ripple(position.xy, vec2f(-0.7, 0.7), 1.0) * 1.0;

    // TODO: use saturate() instead of clamp() once it works
    return 1.0 - clamp(ripples * 10.0, 0.0, 1.0) * 0.5;
}

fn ripple(position: vec2f, vector: vec2f, time_scale: f32) -> f32 {
    const space_scale: f32 = 0.002;
    return sin(dot(position, vector) * space_scale - time * time_scale);
}
