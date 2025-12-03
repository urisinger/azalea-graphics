use std::collections::HashMap;

use azalea_core::direction::Direction;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct ModelPart {
    children: HashMap<String, ModelPart>,
    default_transform: Transform,
    cuboids: Vec<Cuboid>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Vertex {
    pos: glam::Vec3,
    uv: Option<glam::Vec2>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Transform {
    pivot: glam::Vec3,
    rotation: glam::Vec3,
    scale: glam::Vec3,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Cuboid {
    min: glam::Vec3,
    max: glam::Vec3,

    sides: Vec<Side>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Side {
    dir: Direction,
    vertices: Vec<Vertex>,
}
