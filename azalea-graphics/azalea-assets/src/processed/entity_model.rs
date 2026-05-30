use std::collections::HashMap;

use glam::{IVec2, Vec2, Vec3};
pub use raw::Transform;

use super::super::raw::entity_model as raw;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Model {
    pub vertices: Vec<Vertex>,
    pub part: ModelPart,
    pub default_transforms: Vec<Transform>,
}

impl Model {
    pub fn from_raw(raw: raw::LayerDefinition) -> Self {
        let mut vertices = Vec::new();
        let mut default_transforms = Vec::new();
        let texture_size = Vec2::new(raw.texture_size[0] as f32, raw.texture_size[1] as f32);
        let part = ModelPart::from_raw(
            raw.root,
            texture_size,
            &mut vertices,
            &mut default_transforms,
        );
        Self {
            vertices,
            part,
            default_transforms,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelPart {
    pub children: HashMap<String, ModelPart>,
    pub id: usize,
}

impl ModelPart {
    fn from_raw(
        raw: raw::PartDefinition,
        texture_size: Vec2,
        vertices: &mut Vec<Vertex>,
        default_transforms: &mut Vec<Transform>,
    ) -> Self {
        let id = default_transforms.len();
        default_transforms.push(raw.transform);

        for cube in raw.cubes {
            bake_cube(&cube, texture_size, id as u32, vertices);
        }

        let mut children = HashMap::new();
        for (name, child) in raw.children {
            let child_part = ModelPart::from_raw(child, texture_size, vertices, default_transforms);
            children.insert(name, child_part);
        }

        Self { children, id }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Vertex {
    pub pos: Vec3,
    pub uv: Vec2,
    pub transform_id: u32,
}

fn bake_cube(
    cube: &raw::CubeDefinition,
    texture_size: Vec2,

    transform_id: u32,
    vertices: &mut Vec<Vertex>,
) {
    let mut min_x = cube.origin.x;
    let mut min_y = cube.origin.y;
    let mut min_z = cube.origin.z;
    let width = cube.dimensions.x;
    let height = cube.dimensions.y;
    let depth = cube.dimensions.z;

    let mut max_x = min_x + width;
    let mut max_y = min_y + height;
    let mut max_z = min_z + depth;

    let grow_x = cube.grow.x;
    let grow_y = cube.grow.y;
    let grow_z = cube.grow.z;

    min_x -= grow_x;
    min_y -= grow_y;
    min_z -= grow_z;
    max_x += grow_x;
    max_y += grow_y;
    max_z += grow_z;

    if cube.mirror {
        let tmp = max_x;
        max_x = min_x;
        min_x = tmp;
    }

    let t0 = Vec3::new(min_x, min_y, min_z);
    let t1 = Vec3::new(max_x, min_y, min_z);
    let t2 = Vec3::new(max_x, max_y, min_z);
    let t3 = Vec3::new(min_x, max_y, min_z);

    let l0 = Vec3::new(min_x, min_y, max_z);
    let l1 = Vec3::new(max_x, min_y, max_z);
    let l2 = Vec3::new(max_x, max_y, max_z);
    let l3 = Vec3::new(min_x, max_y, max_z);

    let x_tex_offs = cube.tex_coord.x;
    let y_tex_offs = cube.tex_coord.y;

    let u0 = x_tex_offs;
    let u1 = x_tex_offs + depth;
    let u2 = x_tex_offs + depth + width;
    let u22 = x_tex_offs + depth + width + width;
    let u3 = x_tex_offs + depth + width + depth;
    let u4 = x_tex_offs + depth + width + depth + width;

    let v0 = y_tex_offs;
    let v1 = y_tex_offs + depth;
    let v2 = y_tex_offs + depth + height;

    let tex_scale_x = texture_size.x * cube.tex_scale.x;
    let tex_scale_y = texture_size.y * cube.tex_scale.y;

    let mut add_polygon = |mut points: [Vec3; 4], u0: f32, v0: f32, u1: f32, v1: f32| {
        let us = 0.0 / tex_scale_x;
        let vs = 0.0 / tex_scale_y;

        let mut uvs = [
            Vec2::new(u1 / tex_scale_x - us, v0 / tex_scale_y + vs),
            Vec2::new(u0 / tex_scale_x + us, v0 / tex_scale_y + vs),
            Vec2::new(u0 / tex_scale_x + us, v1 / tex_scale_y - vs),
            Vec2::new(u1 / tex_scale_x - us, v1 / tex_scale_y - vs),
        ];

        if cube.mirror {
            points.reverse();
            uvs.reverse();
        }

        // Java: builder.addVertex(pos.x(), pos.y(), pos.z(), ...)
        // worldX() = x / 16.0F

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

    if cube.visible_faces[0] {
        // DOWN
        add_polygon([l1, l0, t0, t1], u1, v0, u2, v1);
    }
    if cube.visible_faces[1] {
        // UP
        add_polygon([t2, t3, l3, l2], u2, v1, u22, v0);
    }
    if cube.visible_faces[4] {
        // WEST
        add_polygon([t0, l0, l3, t3], u0, v1, u1, v2);
    }
    if cube.visible_faces[2] {
        // NORTH
        add_polygon([t1, t0, t3, t2], u1, v1, u2, v2);
    }
    if cube.visible_faces[5] {
        // EAST
        add_polygon([l1, t1, t2, l2], u2, v1, u3, v2);
    }
    if cube.visible_faces[3] {
        // SOUTH
        add_polygon([l0, l1, l2, l3], u3, v1, u4, v2);
    }
}
