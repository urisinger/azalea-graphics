use std::collections::HashMap;

use azalea::core::position::ChunkSectionPos;

use crate::renderer::{
    mesh::Mesh,
    vulkan::{context::VkContext},
};

use super::types::BlockVertex;
use super::staging::StagingArena;
use super::mesher::{Mesher, MeshResult};

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
    pub fn insert_block(&mut self, key: ChunkSectionPos, mesh: Mesh<BlockVertex>) -> Option<Mesh<BlockVertex>> {
        self.blocks.insert(key, mesh)
    }

    pub fn insert_water(&mut self, key: ChunkSectionPos, mesh: Mesh<BlockVertex>) -> Option<Mesh<BlockVertex>> {
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
        staging: &mut StagingArena,
    ) {
        while let Some(MeshResult { blocks, water }) = mesher.as_ref().and_then(|m| m.poll()) {
            if !blocks.vertices.is_empty() {
                let staging_mesh = Mesh::new_staging(ctx, &blocks.vertices, &blocks.indices);
                let mesh = staging_mesh.upload(ctx, cmd);
                staging.push(frame_index, staging_mesh.buffer);

                if let Some(mut old_mesh) = self.insert_block(blocks.section_pos, mesh) {
                    old_mesh.destroy(ctx);
                }
            }

            if !water.vertices.is_empty() {
                let staging_mesh = Mesh::new_staging(ctx, &water.vertices, &water.indices);
                let mesh = staging_mesh.upload(ctx, cmd);
                staging.push(frame_index, staging_mesh.buffer);

                if let Some(mut old_mesh) = self.insert_water(water.section_pos, mesh) {
                    old_mesh.destroy(ctx);
                }
            }
        }
    }
}


