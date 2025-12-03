use std::sync::Arc;

use crate::processed::model::BlockModel;

pub mod animation;
pub mod atlas;
pub mod model;

#[derive(Debug, Clone)]
pub struct VariantDesc {
    pub model: Arc<BlockModel>,

    pub x_rotation: i32,

    pub y_rotation: i32,

    pub uvlock: bool,
}
