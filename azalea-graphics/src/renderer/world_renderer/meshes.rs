use std::collections::HashMap;

use ash::vk;
use azalea::core::position::ChunkSectionPos;

use super::{
    mesher::{MeshResult, Mesher},
    types::BlockVertex,
};
use crate::renderer::{
    mesh::Mesh,
    vulkan::{context::VkContext, frame_sync::FrameSync},
};

pub struct MeshStore {
    pub blocks: HashMap<ChunkSectionPos, Mesh<BlockVertex>>,
    pub water: HashMap<ChunkSectionPos, Mesh<BlockVertex>>,
}

impl Default for MeshStore {
    fn default() -> Self {
        Self {
            blocks: HashMap::new(),
            water: HashMap::new(),
        }
    }
}

impl MeshStore {
    pub fn insert_block(
        &mut self,
        key: ChunkSectionPos,
        mesh: Mesh<BlockVertex>,
    ) -> Option<Mesh<BlockVertex>> {
        self.blocks.insert(key, mesh)
    }

    pub fn insert_water(
        &mut self,
        key: ChunkSectionPos,
        mesh: Mesh<BlockVertex>,
    ) -> Option<Mesh<BlockVertex>> {
        self.water.insert(key, mesh)
    }

    pub fn drain_and_destroy(&mut self, ctx: &VkContext) {
        for (_, mut mesh) in self.blocks.drain() {
            mesh.destroy(ctx);
        }
        for (_, mut mesh) in self.water.drain() {
            mesh.destroy(ctx);
        }
    }

    pub fn process_mesher_results(
        &mut self,
        ctx: &VkContext,
        cmd: ash::vk::CommandBuffer,
        frame_index: usize,
        mesher: &Option<Mesher>,
        frame_sync: &mut FrameSync,
    ) {
        let mut touched_buffers: Vec<vk::Buffer> = Vec::new();

        while let Some(MeshResult { blocks, water }) = mesher.as_ref().and_then(|m| m.poll()) {
            if !blocks.vertices.is_empty() {
                let staging_mesh = Mesh::new_staging(ctx, &blocks.vertices, &blocks.indices);
                let mesh = staging_mesh.upload(ctx, cmd);
                frame_sync.add_to_deletion_queue(frame_index, Box::new(staging_mesh.buffer));

                touched_buffers.push(mesh.buffer.buffer);

                if let Some(mut old_mesh) = self.insert_block(blocks.section_pos, mesh) {
                    old_mesh.destroy(ctx);
                }
            }

            if !water.vertices.is_empty() {
                let staging_mesh = Mesh::new_staging(ctx, &water.vertices, &water.indices);
                let mesh = staging_mesh.upload(ctx, cmd);
                frame_sync.add_to_deletion_queue(frame_index, Box::new(staging_mesh.buffer));

                touched_buffers.push(mesh.buffer.buffer);

                if let Some(mut old_mesh) = self.insert_water(water.section_pos, mesh) {
                    old_mesh.destroy(ctx);
                }
            }
        }

        if !touched_buffers.is_empty() {
            let barriers: Vec<vk::BufferMemoryBarrier> = touched_buffers
                .iter()
                .map(|&buf| {
                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::VERTEX_ATTRIBUTE_READ | vk::AccessFlags::INDEX_READ,
                        )
                        .buffer(buf)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                })
                .collect();

            unsafe {
                ctx.device().cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::VERTEX_INPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &barriers,
                    &[],
                );
            }
        }
    }
}
