use spirv_std::{
    glam::{IVec3, Mat4, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles},
    image::{Image, SampledImage},
    num_traits::Float,
    spirv,
};

#[repr(C)]
pub struct PushConstants {
    pub view_proj: Mat4,
    pub grid_origin_ws: Vec4,
    pub radius: i32,
    pub height: i32,
}

#[spirv(compute(threads(1, 1, 1)))]
pub fn cull_chunks(
    #[spirv(push_constant)] pc: &PushConstants,

    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] visible: &mut [f32],
    #[spirv(descriptor_set = 1, binding = 0)] hiz: &SampledImage<Image!(2D, type=f32, sampled)>,
    #[spirv(global_invocation_id)] gid: IVec3,
) {
    const CHUNK_SIZE: f32 = 16.0;
    let side = pc.radius * 2 + 1;

    if gid.y < 0 || gid.y >= pc.height || gid.x < 0 || gid.x >= side || gid.z < 0 || gid.z >= side {
        return;
    }

    let dx = gid.x - pc.radius;
    let dy = gid.y;
    let dz = gid.z - pc.radius;

    let index = (dy * side * side) + ((dz + pc.radius) * side) + (dx + pc.radius);

    let base =
        pc.grid_origin_ws.truncate() + Vec3::new(dx as f32, dy as f32, dz as f32) * CHUNK_SIZE;
    let bmin = base;
    let bmax = base + Vec3::splat(CHUNK_SIZE);

    let corners_ws: [_; 8] = [
        Vec3::new(bmin.x, bmin.y, bmin.z),
        Vec3::new(bmax.x, bmin.y, bmin.z),
        Vec3::new(bmin.x, bmax.y, bmin.z),
        Vec3::new(bmax.x, bmax.y, bmin.z),
        Vec3::new(bmin.x, bmin.y, bmax.z),
        Vec3::new(bmax.x, bmin.y, bmax.z),
        Vec3::new(bmin.x, bmax.y, bmax.z),
        Vec3::new(bmax.x, bmax.y, bmax.z),
    ];

    let corners_clip: [_; 8] = [
        pc.view_proj * corners_ws[0].extend(1.0),
        pc.view_proj * corners_ws[1].extend(1.0),
        pc.view_proj * corners_ws[2].extend(1.0),
        pc.view_proj * corners_ws[3].extend(1.0),
        pc.view_proj * corners_ws[4].extend(1.0),
        pc.view_proj * corners_ws[5].extend(1.0),
        pc.view_proj * corners_ws[6].extend(1.0),
        pc.view_proj * corners_ws[7].extend(1.0),
    ];

    for plane in 0..6 {
        let mut all_outside = true;
        for i in 0..8 {
            let clip = corners_clip[i];
            let cond = match plane {
                0 => clip.x >= -clip.w, // left
                1 => clip.x <= clip.w,  // right
                2 => clip.y >= -clip.w, // bottom
                3 => clip.y <= clip.w,  // top
                4 => clip.z <= clip.w,  // near
                5 => clip.z >= 0.0,     // far
                _ => false,
            };
            if cond {
                all_outside = false;
                break;
            }
        }
        if all_outside {
            visible[index as usize] = 0.0;
            return;
        }
    }

    let mut min_xy = Vec2::splat(1.0);
    let mut max_xy = Vec2::splat(0.0);
    let mut near_depth = 1.0f32;
    let mut any_valid = false;

    for i in 0..8 {
        let clip = corners_clip[i];
        if clip.w <= 0.0 {
            continue;
        }
        let ndc = clip.truncate() / clip.w;
        let uv = ndc.truncate() * 0.5 + Vec2::splat(0.5);
        let d = ndc.z;

        any_valid = true;
        min_xy = min_xy.min(uv);
        max_xy = max_xy.max(uv);
        near_depth = near_depth.min(d);
    }

    min_xy = min_xy.clamp(Vec2::splat(0.0), Vec2::splat(1.0));
    max_xy = max_xy.clamp(Vec2::splat(0.0), Vec2::splat(1.0));
    if !any_valid {
        visible[index as usize] = 0.0;
        return;
    }

    let extent = max_xy - min_xy;
    if extent.x <= 0.0 || extent.y <= 0.0 {
        visible[index as usize] = 0.0;
        return;
    }

    let tex_size: UVec2 = hiz.query_size_lod(0);
    let texel_size: Vec2 = extent * tex_size.as_vec2();
    let mip = texel_size.x.max(texel_size.y).log2().floor();

    let rect = Vec4::new(min_xy.x, min_xy.y, max_xy.x, max_xy.y);

    let sample1 = hiz.sample_by_lod(rect.xy(), mip).x;
    let sample2 = hiz.sample_by_lod(rect.zy(), mip).x;
    let sample3 = hiz.sample_by_lod(rect.xw(), mip).x;
    let sample4 = hiz.sample_by_lod(rect.zw(), mip).x;
    let max_z = sample1.min(sample2).min(sample3.min(sample4));

    visible[index as usize] = if near_depth > max_z { near_depth } else { 0.0 };
}
