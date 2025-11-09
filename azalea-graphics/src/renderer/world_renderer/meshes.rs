use std::collections::HashMap;

use ash::vk;
use azalea::core::position::ChunkSectionPos;

use super::{
    mesher::{MeshResult, Mesher},
    types::BlockVertex,
};
use crate::renderer::{frame_ctx::FrameCtx, mesh::Mesh, vulkan::context::VkContext};

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

    pub fn process_mesher_results(&mut self, frame_ctx: &mut FrameCtx, mesher: &Option<Mesher>) {
        let mut touched_buffers: Vec<vk::Buffer> = Vec::new();

        while let Some(MeshResult { blocks, water }) = mesher.as_ref().and_then(|m| m.poll()) {
            if !blocks.vertices.is_empty() {
                let staging_mesh =
                    Mesh::new_staging(frame_ctx.ctx, &blocks.vertices, &blocks.indices);
                let mesh = staging_mesh.upload(frame_ctx.ctx, frame_ctx.cmd);
                frame_ctx.delete(staging_mesh.buffer);

                touched_buffers.push(mesh.buffer.buffer);

                if let Some(mut old_mesh) = self.insert_block(blocks.section_pos, mesh) {
                    old_mesh.destroy(frame_ctx.ctx);
                }
            }

            if !water.vertices.is_empty() {
                let staging_mesh =
                    Mesh::new_staging(frame_ctx.ctx, &water.vertices, &water.indices);
                let mesh = staging_mesh.upload(frame_ctx.ctx, frame_ctx.cmd);
                frame_ctx.delete(staging_mesh.buffer);

                touched_buffers.push(mesh.buffer.buffer);

                if let Some(mut old_mesh) = self.insert_water(water.section_pos, mesh) {
                    old_mesh.destroy(frame_ctx.ctx);
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

            frame_ctx.pipeline_barrier(
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                &barriers,
                &[],
            );
        }
    }
}
