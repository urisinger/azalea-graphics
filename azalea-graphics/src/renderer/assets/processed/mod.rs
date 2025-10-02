use std::sync::Arc;

use crate::renderer::assets::processed::model::BlockModel;

pub(crate) mod animation;
pub(crate) mod atlas;
pub(crate) mod model;

#[derive(Debug, Clone)]
pub struct VariantDesc {
    pub model: Arc<BlockModel>,

    pub x_rotation: i32,

    pub y_rotation: i32,

    pub uvlock: bool,
}
