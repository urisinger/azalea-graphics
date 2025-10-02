use spirv_std::{
    glam::{IVec3, Mat4, Vec3, Vec4},
    spirv,
};

#[repr(C)]
pub struct PC {
    pub view_proj: Mat4,
    pub grid_origin_ws: Vec4,
    pub radius: i32,
    pub height: i32,
}

fn chunk_coords(instance: u32, pc: &PC) -> IVec3 {
    let side = pc.radius * 2 + 1;
    let layer_size = side * side;

    let y = (instance as i32) / layer_size;
    let rem = (instance as i32) % layer_size;
    let z = rem / side;
    let x = rem % side;

    IVec3::new(x - pc.radius, y, z - pc.radius)
}

#[spirv(vertex)]
pub fn aabb_vert(
    #[spirv(push_constant)] pc: &PC,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] visible: &[u32],

    #[spirv(vertex_index)] vertex_index: i32,
    #[spirv(instance_index)] instance_index: u32,

    #[spirv(position)] out_pos: &mut Vec4,
    out_color: &mut Vec4,
) {
    let chunk = instance_index;

    if visible[chunk as usize] == 0 {
        *out_pos = Vec4::new(2.0, 2.0, 2.0, 1.0);
        *out_color = Vec4::ZERO;
        return;
    }

    let coord = chunk_coords(chunk, pc);
    let base = pc.grid_origin_ws.truncate() + coord.as_vec3() * 16.0;
    let bmin = base;
    let bmax = base + Vec3::splat(16.0);

    let vidx = match vertex_index {
        0 => 0,
        1 => 1,
        2 => 1,
        3 => 2,
        4 => 2,
        5 => 3,
        6 => 3,
        7 => 0,
        8 => 4,
        9 => 5,
        10 => 5,
        11 => 6,
        12 => 6,
        13 => 7,
        14 => 7,
        15 => 4,
        16 => 0,
        17 => 4,
        18 => 1,
        19 => 5,
        20 => 2,
        21 => 6,
        22 => 3,
        23 => 7,
        _ => 0,
    };

    let unit = match vidx {
        0 => Vec3::new(0.0, 0.0, 0.0),
        1 => Vec3::new(1.0, 0.0, 0.0),
        2 => Vec3::new(1.0, 1.0, 0.0),
        3 => Vec3::new(0.0, 1.0, 0.0),
        4 => Vec3::new(0.0, 0.0, 1.0),
        5 => Vec3::new(1.0, 0.0, 1.0),
        6 => Vec3::new(1.0, 1.0, 1.0),
        7 => Vec3::new(0.0, 1.0, 1.0),
        _ => Vec3::ZERO,
    };

    let world = bmin + (bmax - bmin) * unit;

    *out_pos = pc.view_proj * world.extend(1.0);
    *out_color = Vec4::new(1.0, 0.0, 0.0, 1.0);
}

#[spirv(fragment)]
#[unsafe(no_mangle)]
pub fn aabb_frag(in_color: Vec4, frag_color: &mut Vec4) {
    *frag_color = in_color;
}
