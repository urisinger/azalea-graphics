use ash::vk;
use azalea::core::position::ChunkSectionPos;

use crate::renderer::world_renderer::{meshes::MeshStore, pipelines::Pipelines, types::PushConstants, visibility};

pub fn draw_world(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    pipelines: &Pipelines,
    descriptor_set: vk::DescriptorSet,
    meshes: &MeshStore,
    view_proj: glam::Mat4,
    wireframe_mode: bool,
    camera_pos: glam::Vec3,
) {
    let push = PushConstants { view_proj };

    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipelines.block_pipeline(wireframe_mode));
        device.cmd_push_constants(
            cmd,
            pipelines.layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            std::slice::from_raw_parts(&push as *const PushConstants as *const u8, std::mem::size_of::<PushConstants>()),
        );
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipelines.layout,
            0,
            &[descriptor_set],
            &[],
        );
    }

    for (pos, mesh) in &meshes.blocks {
        let pos_min = glam::Vec3::new(pos.x as f32 * 16.0, pos.y as f32 * 16.0, pos.z as f32 * 16.0);
        let pos_max = glam::Vec3::new(pos_min.x + 16.0, pos_min.y + 16.0, pos_min.z + 16.0);
        if !visibility::aabb_visible(view_proj, pos_min, pos_max) {
            continue;
        }
        let vertex_buffers = [mesh.buffer.buffer];
        let offsets = [mesh.vertex_offset];
        unsafe {
            device.cmd_bind_vertex_buffers(cmd, 0, &vertex_buffers, &offsets);
            device.cmd_bind_index_buffer(cmd, mesh.buffer.buffer, mesh.index_offset, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
        }
    }

    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipelines.water_pipeline(wireframe_mode));
        device.cmd_push_constants(
            cmd,
            pipelines.layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            std::slice::from_raw_parts(&push as *const PushConstants as *const u8, std::mem::size_of::<PushConstants>()),
        );
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            pipelines.layout,
            0,
            &[descriptor_set],
            &[],
        );
    }

    let mut water_meshes: Vec<_> = meshes.water.iter().collect();
    water_meshes.sort_by(|(a, _), (b, _)| {
        let dist = |pos: &ChunkSectionPos| {
            camera_pos.distance_squared(glam::Vec3::new(
                pos.x as f32 * 16.0 + 8.0,
                pos.y as f32 * 16.0 + 8.0,
                pos.z as f32 * 16.0 + 8.0,
            ))
        };
        dist(a).partial_cmp(&dist(b)).unwrap_or(std::cmp::Ordering::Equal)
    });

    for (_, mesh) in water_meshes {
        let vertex_buffers = [mesh.buffer.buffer];
        let offsets = [mesh.vertex_offset];
        unsafe {
            device.cmd_bind_vertex_buffers(cmd, 0, &vertex_buffers, &offsets);
            device.cmd_bind_index_buffer(cmd, mesh.buffer.buffer, mesh.index_offset, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
        }
    }
}


