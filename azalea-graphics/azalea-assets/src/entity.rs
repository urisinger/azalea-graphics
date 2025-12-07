use std::collections::HashMap;

use azalea_core::direction::Direction;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct ModelPart {
    pub children: HashMap<String, ModelPart>,
    pub default_transform: Transform,
    pub cuboids: Vec<Cuboid>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Vertex {
    pub pos: glam::Vec3,
    pub uv: Option<glam::Vec2>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Transform {
    pub pivot: glam::Vec3,
    pub rotation: glam::Vec3,
    pub scale: glam::Vec3,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Cuboid {
    pub min: glam::Vec3,
    pub max: glam::Vec3,

    pub sides: Vec<Side>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Side {
    pub dir: Direction,
    pub vertices: Vec<Vertex>,
}
