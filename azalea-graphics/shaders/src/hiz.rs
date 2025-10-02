use spirv_std::{
    glam::{UVec2, UVec3, Vec4},
    image::Image,
    spirv,
};

#[spirv(compute(threads(8, 8, 1)))]
pub fn copy(
    #[spirv(descriptor_set = 0, binding = 0)] src: &Image!(2D, type=f32, sampled),

    #[spirv(descriptor_set = 0, binding = 1)] dst: &Image!(2D, format = r32f, sampled = false),

    #[spirv(global_invocation_id)] id: UVec3,
) {
    let dst_size: UVec2 = dst.query_size();
    if id.x >= dst_size.x || id.y >= dst_size.y {
        return;
    }

    let d = src.fetch(id.truncate());

    unsafe { dst.write(id.truncate(), d) };
}

#[spirv(compute(threads(8, 8, 1)))]
pub fn reduce(
    #[spirv(descriptor_set = 0, binding = 0)] src: &Image!(2D, format = r32f, sampled = false),
    #[spirv(descriptor_set = 0, binding = 1)] dst: &Image!(2D, format = r32f, sampled = false),
    #[spirv(global_invocation_id)] id: UVec3,
) {
    let dst_size: UVec2 = dst.query_size();
    let o = id.truncate();
    if o.x >= dst_size.x || o.y >= dst_size.y {
        return;
    }

    let src_size: UVec2 = src.query_size();
    let base = o * 2;

    let p00 = base.min(src_size - 1);
    let p10 = (base + UVec2::new(1, 0)).min(src_size - 1);
    let p01 = (base + UVec2::new(0, 1)).min(src_size - 1);
    let p11 = (base + UVec2::new(1, 1)).min(src_size - 1);

    let d0 = src.read(p00);
    let d1 = src.read(p10);
    let d2 = src.read(p01);
    let d3 = src.read(p11);

    let d = d0.min(d1).min(d2.min(d3));
    unsafe { dst.write(o, Vec4::new(d, 0.0, 0.0, 0.0)) };
}
