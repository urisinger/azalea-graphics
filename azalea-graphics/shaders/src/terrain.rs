use spirv_std::{
    arch::kill,
    glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles},
    image::{Image, SampledImage},
    spirv,
};

#[repr(C)]
pub struct WorldUniform {
    pub view_proj: Mat4,
}

#[spirv(vertex)]
pub fn block_vert(
    #[spirv(descriptor_set = 0, binding = 1, uniform)] pc: &WorldUniform,

    in_pos: Vec3,
    in_ao: f32,
    in_uv: Vec2,
    in_tint: Vec3,

    out_uv: &mut Vec2,
    out_ao: &mut f32,
    out_tint: &mut Vec3,

    #[spirv(position)] out_pos: &mut Vec4,
) {
    *out_pos = pc.view_proj * in_pos.extend(1.0);
    *out_uv = in_uv;
    *out_ao = in_ao / 3.0;
    *out_tint = in_tint;
}

#[spirv(fragment)]
pub fn block_frag(
    in_uv: Vec2,
    in_ao: f32,
    in_tint: Vec3,
    #[spirv(descriptor_set = 0, binding = 0)] block_atlas: &SampledImage<
        Image!(2D, type=f32, sampled),
    >,
    frag_color: &mut Vec4,
) {
    let tex_color: Vec4 = block_atlas.sample(in_uv);
    if tex_color.w < 0.1 {
        kill()
    }

    *frag_color = (tex_color.xyz() * in_tint * in_ao).extend(tex_color.w);
}

#[spirv(vertex)]
pub fn water_vert(
    #[spirv(descriptor_set = 0, binding = 1, uniform)] pc: &WorldUniform,

    in_pos: Vec3,
    in_ao: f32,
    in_uv: Vec2,
    in_tint: Vec3,

    out_uv: &mut Vec2,
    out_ao: &mut f32,
    out_tint: &mut Vec3,

    #[spirv(position)] clip_pos: &mut Vec4,
) {
    *clip_pos = pc.view_proj * in_pos.extend(1.0);
    *out_uv = in_uv;
    *out_ao = in_ao / 3.0;
    *out_tint = in_tint;
}

#[spirv(fragment)]
pub fn water_frag(
    in_uv: Vec2,
    in_ao: f32,
    in_tint: Vec3,
    #[spirv(descriptor_set = 0, binding = 0)] block_atlas: &SampledImage<
        Image!(2D, type=f32, sampled),
    >,
    frag_color: &mut Vec4,
) {
    let tex_color: Vec4 = block_atlas.sample(in_uv);
    *frag_color = (tex_color.xyz() * in_tint * in_ao).extend(tex_color.w);
}
