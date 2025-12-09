use spirv_std::{
    Image, RuntimeArray,
    glam::{Mat4, Vec2, Vec3, Vec4},
    image::SampledImage,
    spirv,
};

use crate::terrain::WorldUniform;

#[repr(C)]
pub struct PC {
    texture: u32,
    transform_offset: u32,
}
#[spirv(vertex)]
pub fn vert(
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniform: &WorldUniform,
    #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] transforms: &[Mat4],
    #[spirv(push_constant)] pc: &PC,

    in_pos: Vec3,
    in_transform_id: u32,
    in_uv: Vec2,

    out_uv: &mut Vec2,
    out_texture: &mut u32,

    #[spirv(position)] out_pos: &mut Vec4,
) {
    *out_pos = uniform.view_proj * transforms[in_transform_id as usize + pc.transform_offset as usize] * in_pos.extend(1.0);
    *out_uv = in_uv;
    *out_texture = pc.texture;
}

#[spirv(fragment)]
pub fn frag(
    in_uv: Vec2,
    #[spirv(flat)] in_tex: u32,
    #[spirv(descriptor_set = 1, binding = 0)] textures: &RuntimeArray<
        SampledImage<Image!(2D, type=f32, sampled)>,
    >,
    frag_color: &mut Vec4,
) {
    let tex_color: Vec4 = unsafe { textures.index(in_tex as usize).sample(in_uv) };
    *frag_color = tex_color;
}
