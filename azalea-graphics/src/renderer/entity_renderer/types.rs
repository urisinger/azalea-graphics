use std::mem::offset_of;

use ash::vk;
use glam::{Mat4, Vec2, Vec3};

pub struct EntityVertex {
    pub pos: Vec3,
    pub transform_id: u32,
    pub uv: Vec2,
}

impl EntityVertex {
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<EntityVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription] {
        &[
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(EntityVertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32_UINT,
                offset: offset_of!(EntityVertex, transform_id) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(EntityVertex, uv) as u32,
            },
        ]
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EntityPushConstants {
    pub model: Mat4,
    pub tex_id: u32,
}
