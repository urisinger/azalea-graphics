use spirv_std::{
    glam::{Vec2, Vec4},
    image::{Image, SampledImage},
    spirv,
};

#[repr(C)]
pub struct BlockPushConstants {
    screen_size: Vec2,
}

#[spirv(vertex)]
pub fn egui_vert(
    #[spirv(push_constant)] pc: &BlockPushConstants,
    in_pos: Vec2,
    in_uv: Vec2,
    in_rgba: Vec4,

    out_rgba: &mut Vec4,
    out_uv: &mut Vec2,

    #[spirv(position)] out_pos: &mut Vec4,
) {
    *out_pos = Vec4::new(
        2.0 * in_pos.x / pc.screen_size.x - 1.0,
        2.0 * in_pos.y / pc.screen_size.y - 1.0,
        0.0,
        1.0,
    );
    *out_rgba = in_rgba;
    *out_uv = in_uv;
}

#[spirv(fragment)]
pub fn egui_frag(
    #[spirv(descriptor_set = 0, binding = 0)] texture: &SampledImage<Image!(2D, type=f32, sampled)>,

    in_rgba: Vec4,
    in_uv: Vec2,

    frag_color: &mut Vec4,
) {
    let tex_color = texture.sample(in_uv);

    *frag_color = in_rgba * tex_color;
}
