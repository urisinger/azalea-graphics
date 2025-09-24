use std::{collections::HashSet, io::Cursor, sync::Arc, thread};

use azalea::{
    blocks::BlockState,
    core::{
        position::{ChunkPos, ChunkSectionPos},
        registry_holder::{BiomeData, RegistryHolder},
    },
    registry::{Biome, Block, DataRegistry},
};
use crossbeam::channel::{Receiver, Sender, unbounded};
use glam::IVec3;
use log::error;
use parking_lot::{RawMutex, RwLock, lock_api::Mutex};
use simdnbt::Deserialize;

use crate::renderer::{
    assets::Assets,
    chunk::{LocalChunk, LocalSection},
    world_renderer::{
        BlockVertex,
        mesher::{block::mesh_block, water::mesh_water},
    },
};

mod block;
mod block_colors;
mod helpers;
mod water;

pub struct MeshData {
    pub vertices: Vec<BlockVertex>,
    pub indices: Vec<u32>,
    pub section_pos: ChunkSectionPos,
}

pub struct Mesher {
    work_tx: Sender<ChunkSectionPos>,
    result_rx: Receiver<MeshResult>,

    pub world: Arc<RwLock<azalea::world::Instance>>,
    dirty: Arc<Mutex<RawMutex, HashSet<ChunkSectionPos>>>,
}

impl Mesher {
    pub fn new(
        assets: Arc<Assets>,
        world: Arc<RwLock<azalea::world::Instance>>,
        num_threads: usize,
    ) -> Self {
        let (work_tx, work_rx) = unbounded::<ChunkSectionPos>();
        let (result_tx, result_rx) = unbounded::<MeshResult>();
        let dirty = Arc::new(Mutex::new(HashSet::new()));
        let biome_cache = Arc::new(BiomeCache::from_registries(&world.read().registries));

        for id in 0..num_threads {
            let work_rx = work_rx.clone();
            let result_tx = result_tx.clone();
            let assets = assets.clone();
            let dirty = dirty.clone();
            let world = world.clone();
            let biome_cache = biome_cache.clone();

            thread::spawn(move || {
                while let Ok(spos) = work_rx.recv() {
                    dirty.lock().remove(&spos);
                    if let Some(local) = build_local_section(&world, spos) {
                        let start = std::time::Instant::now();
                        let mesh = mesh_section(&local, &biome_cache, &assets);
                        result_tx.send(mesh).unwrap();

                        let dt = start.elapsed();
                    }
                }
            });
        }

        Self {
            work_tx,
            result_rx,
            world,
            dirty,
        }
    }

    pub fn submit_section(&self, spos: ChunkSectionPos) {
        let mut dirty = self.dirty.lock();
        if dirty.insert(spos) {
            self.work_tx.send(spos).unwrap();
        }
    }

    pub fn submit_chunk(&self, pos: ChunkPos) {

    }

    pub fn poll(&self) -> Option<MeshResult> {
        self.result_rx.try_recv().ok()
    }
}

fn build_local_section(
    world: &Arc<RwLock<azalea::world::Instance>>,
    spos: ChunkSectionPos,
) -> Option<LocalSection> {
    let world_guard = world.read();

    let center = world_guard
        .chunks
        .get(&ChunkPos::new(spos.x, spos.z))?
        .clone();

    let neighbors = [
        world_guard.chunks.get(&ChunkPos::new(spos.x, spos.z - 1)), // North
        world_guard.chunks.get(&ChunkPos::new(spos.x, spos.z + 1)), // South
        world_guard.chunks.get(&ChunkPos::new(spos.x + 1, spos.z)), // East
        world_guard.chunks.get(&ChunkPos::new(spos.x - 1, spos.z)), // West
        world_guard
            .chunks
            .get(&ChunkPos::new(spos.x + 1, spos.z - 1)), // NE
        world_guard
            .chunks
            .get(&ChunkPos::new(spos.x - 1, spos.z - 1)), // NW
        world_guard
            .chunks
            .get(&ChunkPos::new(spos.x + 1, spos.z + 1)), // SE
        world_guard
            .chunks
            .get(&ChunkPos::new(spos.x - 1, spos.z + 1)), // SW
    ];

    drop(world_guard);

    let local_chunk = LocalChunk { center, neighbors };

    Some(local_chunk.borrow_chunks().build_local_section(spos))
}

pub struct MeshResult {
    pub blocks: MeshData,
    pub water: MeshData,
}

pub struct MeshBuilder<'a> {
    pub assets: &'a Assets,
    pub block_colors: &'a block_colors::BlockColors,
    pub section: &'a LocalSection,

    pub biome_cache: &'a BiomeCache,

    block_vertices: Vec<BlockVertex>,
    block_indices: Vec<u32>,
    water_vertices: Vec<BlockVertex>,
    water_indices: Vec<u32>,
}

impl<'a> MeshBuilder<'a> {
    fn block_state_at(&self, pos: IVec3) -> Option<BlockState> {
        self.section.blocks[pos.x as usize][pos.y as usize][pos.z as usize]
    }

    pub fn push_block_quad(&mut self, verts: [BlockVertex; 4]) {
        let start = self.block_vertices.len() as u32;
        self.block_vertices.extend_from_slice(&verts);
        self.block_indices.extend_from_slice(&[
            start,
            start + 1,
            start + 2,
            start,
            start + 2,
            start + 3,
        ]);
    }

    pub fn push_water_quad(&mut self, verts: [BlockVertex; 4]) {
        let start = self.water_vertices.len() as u32;
        self.water_vertices.extend_from_slice(&verts);
        self.water_indices.extend_from_slice(&[
            start,
            start + 1,
            start + 2,
            start,
            start + 2,
            start + 3,
        ]);
    }

    pub fn finish(self) -> MeshResult {
        MeshResult {
            blocks: MeshData {
                section_pos: self.section.spos,
                vertices: self.block_vertices,
                indices: self.block_indices,
            },
            water: MeshData {
                section_pos: self.section.spos,
                vertices: self.water_vertices,
                indices: self.water_indices,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct BiomeCache {
    pub biomes: Vec<BiomeData>,
}

impl BiomeCache {
    fn from_registries(registries: &RegistryHolder) -> Self {
        let mut biomes = Vec::new();

        if let Some(biome_registry) = registries
            .map
            .get(&azalea::ResourceLocation::new(Biome::NAME))
        {
            for (_key, value) in biome_registry {
                let mut nbt_bytes = Vec::new();
                value.write(&mut nbt_bytes);

                let nbt_borrow_compound =
                    match simdnbt::borrow::read_compound(&mut Cursor::new(&nbt_bytes)) {
                        Ok(compound) => compound,
                        Err(e) => {
                            error!("Failed to read NBT compound for biome: {}", e);
                            continue;
                        }
                    };

                let biome_data = match BiomeData::from_compound((&nbt_borrow_compound).into()) {
                    Ok(value) => value,
                    Err(e) => {
                        error!("Failed to parse BiomeData: {}, {value:?}", e);
                        continue;
                    }
                };

                biomes.push(biome_data);
            }
        }

        BiomeCache { biomes }
    }
}

pub fn mesh_section(
    section: &LocalSection,
    biome_cache: &BiomeCache,
    assets: &Assets,
) -> MeshResult {
    let block_colors = block_colors::BlockColors::create_default();

    let mut builder = MeshBuilder {
        assets,
        block_colors: &block_colors,
        section,
        biome_cache,
        block_vertices: Vec::with_capacity(1000),
        block_indices: Vec::with_capacity(1000),
        water_vertices: Vec::with_capacity(500),
        water_indices: Vec::with_capacity(500),
    };

    for y in 0..16 {
        for x in 0..16 {
            for z in 0..16 {
                let local = IVec3::new(x + 1, y + 1, z + 1);
                let block = section.blocks[local.x as usize][local.y as usize][local.z as usize]
                    .unwrap_or(BlockState::AIR);

                if !block.is_air() {
                    if Block::from(block) == Block::Water {
                        mesh_water(block, local, &mut builder);
                    }

                    mesh_block(block, local, &mut builder);
                }
            }
        }
    }

    builder.finish()
}
