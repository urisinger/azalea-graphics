use std::mem::offset_of;

use ash::vk;
use bytemuck::{NoUninit, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockVertex {
    pub position: [f32; 3],
    pub ao: f32,
    pub uv: [f32; 2],
    pub tint: [f32; 3],
}

impl BlockVertex {
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<BlockVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn attribute_descriptions() -> &'static [vk::VertexInputAttributeDescription] {
        &[
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(BlockVertex, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32_SFLOAT,
                offset: offset_of!(BlockVertex, ao) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(BlockVertex, uv) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 3,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(BlockVertex, tint) as u32,
            },
        ]
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PushConstants {
    pub view_proj: glam::Mat4,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Default, Zeroable, NoUninit)]
pub struct VisibilityPushConstants {
    pub view_proj: [[f32; 4]; 4],
    pub grid_origin_ws: [f32; 4],
    pub radius: i32,
    pub height: i32,
    pub _padding: [i32; 2],
}
