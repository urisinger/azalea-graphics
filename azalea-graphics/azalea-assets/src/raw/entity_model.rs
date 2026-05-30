use std::collections::BTreeMap;

use glam::{Mat4, Quat, Vec2, Vec3};

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
            glam::EulerRot::ZYX,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
        ));

        let scale = Mat4::from_scale(self.scale);

        translation * rotation * scale
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CubeDefinition {
    pub comment: Option<String>,
    pub origin: Vec3,
    pub dimensions: Vec3,
    pub grow: Vec3,
    pub mirror: bool,
    pub tex_coord: Vec2,
    pub tex_scale: Vec2,
    pub visible_faces: [bool; 6],
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PartDefinition {
    pub cubes: Vec<CubeDefinition>,
    pub transform: Transform,
    pub children: BTreeMap<String, PartDefinition>,
}
