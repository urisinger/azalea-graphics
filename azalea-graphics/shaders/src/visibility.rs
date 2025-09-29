use spirv_std::{
    glam::{UVec2, UVec3},
    image::{Image},
    spirv,
};

#[spirv(compute(threads(8, 8, 1)))]
pub fn hiz_copy(
    #[spirv(descriptor_set = 0, binding = 0)] src: &Image!(2D, type=f32, sampled=true, depth),

    #[spirv(descriptor_set = 0, binding = 1)] dst: &Image!(2D, format = r32f, sampled=false),

    #[spirv(global_invocation_id)] id: UVec3,
) {
    let dst_size: UVec2 = dst.query_size();
    if id.x >= dst_size.x || id.y >= dst_size.y {
        return;
    }

    let d = src.fetch(id.truncate());

    unsafe { dst.write(id.truncate(), d) };
}
