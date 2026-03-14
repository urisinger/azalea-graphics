use std::collections::HashMap;

use azalea_core::direction::Direction;
use glam::{Mat4, Quat};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelPart {
    pub children: HashMap<String, ModelPart>,
    pub default_transform: Transform,
    pub cuboids: Vec<Cuboid>,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub struct Vertex {
    pub pos: glam::Vec3,
    pub uv: Option<glam::Vec2>,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize)]
pub struct Transform {
    pub pivot: glam::Vec3,
    pub rotation: glam::Vec3,
    pub scale: glam::Vec3,
}

impl Transform {
    pub fn to_mat4(self) -> Mat4 {
        let translation = Mat4::from_translation(self.pivot);

        let rotation = Mat4::from_quat(Quat::from_euler(
            glam::EulerRot::XYZ,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
        ));

        let scale = Mat4::from_scale(self.scale);

        translation * rotation * scale
    }
}



#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Cuboid {
    pub min: glam::Vec3,
    pub max: glam::Vec3,

    pub sides: Vec<Side>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Side {
    pub dir: Direction,
    pub vertices: Vec<Vertex>,
}
