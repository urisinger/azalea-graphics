use std::collections::HashMap;

use glam::{Vec2, Vec3};
pub use raw::Transform;

use super::super::raw::entity_model as raw;

#[derive(Debug, Clone)]
pub struct Model {
    pub vertices: Vec<Vertex>,
    pub part: ModelPart,
    pub default_transforms: Vec<Transform>,
}

impl Model {
    pub fn from_raw(raw: raw::PartDefinition) -> Self {
        let mut vertices = Vec::new();
        let mut default_transforms = Vec::new();
        let part = ModelPart::from_raw(raw, &mut vertices, &mut default_transforms);
        Self {
            vertices,
            part,
            default_transforms,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelPart {
    pub children: HashMap<String, ModelPart>,
    pub id: usize,
}

impl ModelPart {
    fn from_raw(
        raw: raw::PartDefinition,
        vertices: &mut Vec<Vertex>,
        default_transforms: &mut Vec<Transform>,
    ) -> Self {
        let id = default_transforms.len();
        default_transforms.push(raw.transform);

        for cube in raw.cubes {
            bake_cube(&cube, id as u32, vertices);
        }

        let mut children = HashMap::new();
        for (name, child) in raw.children {
            let child_part = ModelPart::from_raw(child, vertices, default_transforms);
            children.insert(name, child_part);
        }

        Self { children, id }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub pos: Vec3,
    pub uv: Vec2,
    pub transform_id: u32,
}

fn bake_cube(cube: &raw::CubeDefinition, transform_id: u32, vertices: &mut Vec<Vertex>) {
    let mut min_x = cube.origin.x;
    let mut min_y = cube.origin.y;
    let mut min_z = cube.origin.z;
    let mut max_x = min_x + cube.dimensions.x;
    let mut max_y = min_y + cube.dimensions.y;
    let mut max_z = min_z + cube.dimensions.z;

    min_x -= cube.grow.x;
    min_y -= cube.grow.y;
    min_z -= cube.grow.z;
    max_x += cube.grow.x;
    max_y += cube.grow.y;
    max_z += cube.grow.z;

    if cube.mirror {
        std::mem::swap(&mut min_x, &mut max_x);
    }

    let t0 = Vec3::new(min_x, min_y, min_z);
    let t1 = Vec3::new(max_x, min_y, min_z);
    let t2 = Vec3::new(max_x, max_y, min_z);
    let t3 = Vec3::new(min_x, max_y, min_z);
    let l0 = Vec3::new(min_x, min_y, max_z);
    let l1 = Vec3::new(max_x, min_y, max_z);
    let l2 = Vec3::new(max_x, max_y, max_z);
    let l3 = Vec3::new(min_x, max_y, max_z);

    let u_offs = cube.tex_coord.x;
    let v_offs = cube.tex_coord.y;
    let width = cube.dimensions.x;
    let height = cube.dimensions.y;
    let depth = cube.dimensions.z;

    let u0 = u_offs;
    let u1 = u_offs + depth;
    let u2 = u_offs + depth + width;
    let u22 = u_offs + depth + width + width;
    let u3 = u_offs + depth + width + depth;
    let u4 = u_offs + depth + width + depth + width;

    let v0 = v_offs;
    let v1 = v_offs + depth;
    let v2 = v_offs + depth + height;

    let tex_scale_x = cube.tex_scale.x;
    let tex_scale_y = cube.tex_scale.y;

    let mut add_quad = |mut points: [Vec3; 4], u0: f32, v0: f32, u1: f32, v1: f32| {
        let mut uvs = [
            Vec2::new(u1 / tex_scale_x, v0 / tex_scale_y),
            Vec2::new(u0 / tex_scale_x, v0 / tex_scale_y),
            Vec2::new(u0 / tex_scale_x, v1 / tex_scale_y),
            Vec2::new(u1 / tex_scale_x, v1 / tex_scale_y),
        ];

        if cube.mirror {
            points.reverse();
            uvs.reverse();
        }

        // triangle 1
        vertices.push(Vertex {
            pos: points[0],
            uv: uvs[0],
            transform_id,
        });
        vertices.push(Vertex {
            pos: points[1],
            uv: uvs[1],
            transform_id,
        });
        vertices.push(Vertex {
            pos: points[2],
            uv: uvs[2],
            transform_id,
        });

        // triangle 2
        vertices.push(Vertex {
            pos: points[0],
            uv: uvs[0],
            transform_id,
        });
        vertices.push(Vertex {
            pos: points[2],
            uv: uvs[2],
            transform_id,
        });
        vertices.push(Vertex {
            pos: points[3],
            uv: uvs[3],
            transform_id,
        });
    };

    // DOWN (0)
    if cube.visible_faces[0] {
        add_quad([l1, l0, t0, t1], u1, v0, u2, v1);
    }
    // UP (1)
    if cube.visible_faces[1] {
        add_quad([t2, t3, l3, l2], u2, v1, u22, v0);
    }
    // WEST (4)
    if cube.visible_faces[4] {
        add_quad([t0, l0, l3, t3], u0, v1, u1, v2);
    }
    // NORTH (2)
    if cube.visible_faces[2] {
        add_quad([t1, t0, t3, t2], u1, v1, u2, v2);
    }
    // EAST (5)
    if cube.visible_faces[5] {
        add_quad([l1, t1, t2, l2], u2, v1, u3, v2);
    }
    // SOUTH (3)
    if cube.visible_faces[3] {
        add_quad([l0, l1, l2, l3], u3, v1, u4, v2);
    }
}
