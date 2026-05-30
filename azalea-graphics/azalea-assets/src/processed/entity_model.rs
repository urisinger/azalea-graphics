use std::collections::HashMap;

use glam::{Vec2, Vec3};
pub use raw::Transform;

use super::super::raw::entity_model as raw;

#[derive(Debug)]
pub struct Model {
    pub vertices: Vec<Vertex>,
    pub part: ModelPart,
    pub default_transforms: Vec<Transform>,
}

impl Model {
    pub fn from_raw(raw: raw::ModelPart) -> Self {
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

#[derive(Debug)]
pub struct ModelPart {
    pub children: HashMap<String, ModelPart>,

    pub id: usize,
}

impl ModelPart {
    fn from_raw(
        raw: raw::ModelPart,
        vertices: &mut Vec<Vertex>,
        default_transforms: &mut Vec<Transform>,
    ) -> Self {
        let id = default_transforms.len();
        default_transforms.push(raw.default_transform);

        for cuboid in raw.cuboids {
            for side in cuboid.sides {
                let v = &side.vertices;

                for i in [0, 1, 2] {
                    vertices.push(Vertex {
                        pos: v[i].pos,
                        uv: v[i].uv.unwrap_or(Vec2::ZERO),
                        transform_id: id as u32,
                    });
                }

                for i in [0, 2, 3] {
                    vertices.push(Vertex {
                        pos: v[i].pos,
                        uv: v[i].uv.unwrap_or(Vec2::ZERO),
                        transform_id: id as u32,
                    });
                }
            }
        }

        let mut children = HashMap::new();
        for (name, child) in raw.children {
            let child_part = ModelPart::from_raw(child, vertices, default_transforms);
            children.insert(name, child_part);
        }

        Self { children, id }
    }
}

#[derive(Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub uv: Vec2,
    pub transform_id: u32,
}
