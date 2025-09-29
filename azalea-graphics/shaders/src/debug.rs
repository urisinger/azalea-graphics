use spirv_std::{glam::{IVec3, Mat4, Vec3, Vec4}, spirv, RuntimeArray};

#[repr(C)]
pub struct PC {
    pub view_proj: Mat4,
    pub grid_origin_ws: Vec4,
    pub radius: i32,
    pub height: i32,
}

#[repr(C)]
pub struct Visibility {
    pub visible: [u32],
}

const CUBE_VERTS: [Vec3; 8] = [
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(0.0, 1.0, 1.0),
];

const CUBE_INDICES: [i32; 24] = [
    0, 1, 1, 2, 2, 3, 3, 0, // bottom
    4, 5, 5, 6, 6, 7, 7, 4, // top
    0, 4, 1, 5, 2, 6, 3, 7, // verticals
];

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
pub fn aabb_debug_vert(
    #[spirv(push_constant)] pc: &PC,
    #[spirv(descriptor_set = 0, binding = 0)] visible: &RuntimeArray<u32>,

    #[spirv(vertex_index)] vertex_index: i32,
    #[spirv(instance_index)] instance_index: u32,

    #[spirv(position)] out_pos: &mut Vec4,
    #[spirv(location = 0)] out_color: &mut Vec3,
) {
    let chunk = instance_index;

    if unsafe { *visible.index(chunk as usize) } == 0 {
        *out_pos = Vec4::new(2.0, 2.0, 2.0, 1.0);
        *out_color = Vec3::ZERO;
        return;
    }

    let coord = chunk_coords(chunk, pc);

    let base = pc.grid_origin_ws.truncate() + coord.as_vec3() * 16.0;
    let bmin = base;
    let bmax = base + Vec3::splat(16.0);

    let vidx = CUBE_INDICES[vertex_index as usize];
    let unit = CUBE_VERTS[vidx as usize];

    let world = bmin.lerp(bmax, unit);

    *out_pos = pc.view_proj * world.extend(1.0);
    *out_color = Vec3::ONE;
}
