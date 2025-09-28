use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashSet},
    io::Cursor,
    sync::Arc,
    thread,
};

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
        visibility::buffers::VisibilitySnapshot,
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

    visibility_tx: Sender<VisibilitySnapshot>,

    pub world: Arc<RwLock<azalea::world::Instance>>,
    dirty: Arc<Mutex<RawMutex, HashSet<ChunkSectionPos>>>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Job {
    prio: u32,
    spos: ChunkSectionPos,
}

impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        other.prio.cmp(&self.prio)
    }
}

impl PartialOrd for Job {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn prio_for(vis: &VisibilitySnapshot, spos: ChunkSectionPos) -> u32 {
    let dx = (spos.x - vis.cx) as i32;
    let dy = (spos.y - vis.cy) as i32;
    let dz = (spos.z - vis.cz) as i32;
    (dx * dx + dz * dz + dy * dy) as u32
}

impl Mesher {
    pub fn new(assets: Arc<Assets>, world: Arc<RwLock<azalea::world::Instance>>) -> Self {
        let (work_tx, work_rx) = unbounded::<ChunkSectionPos>();
        let (result_tx, result_rx) = unbounded::<MeshResult>();
        let (visibility_tx, visibility_rx) = unbounded::<VisibilitySnapshot>();

        let dirty = Arc::new(Mutex::new(HashSet::new()));
        let biome_cache = Arc::new(BiomeCache::from_registries(&world.read().registries));

        let thread_world = Arc::clone(&world);
        let thread_dirty = Arc::clone(&dirty);
        let thread_assets = Arc::clone(&assets);
        let thread_biome_cache = Arc::clone(&biome_cache);

        std::thread::spawn(move || {
            use std::collections::{BinaryHeap, HashSet};

            let mut current_visibility: Option<VisibilitySnapshot> = None;
            let mut queue: BinaryHeap<Job> = BinaryHeap::new();
            let mut queued: HashSet<ChunkSectionPos> = HashSet::new();

            fn push_job(
                queue: &mut BinaryHeap<Job>,
                queued: &mut HashSet<ChunkSectionPos>,
                vis: Option<&VisibilitySnapshot>,
                spos: ChunkSectionPos,
            ) {
                if !queued.insert(spos) {
                    return;
                }
                let prio = vis.map(|v| prio_for(v, spos)).unwrap_or(u32::MAX);
                queue.push(Job { prio, spos });
            }

            loop {
                while let Some(job) = queue.pop() {
                    queued.remove(&job.spos);

                    if let Some(vis) = &current_visibility {
                        if !vis.section_is_visible(job.spos) {
                            continue;
                        }
                    }

                    {
                        let mut d = thread_dirty.lock();
                        if !d.remove(&job.spos) {
                            continue;
                        }
                    }

                    if let Some(local) = build_local_section(&thread_world, job.spos) {
                        let t0 = std::time::Instant::now();
                        let mesh = mesh_section(&local, &thread_biome_cache, &thread_assets);
                        result_tx.send(mesh).unwrap();
                        log::debug!("Meshed {:?} in {:?}", job.spos, t0.elapsed());
                    }
                    if !work_rx.is_empty() || !visibility_rx.is_empty() {
                        break;
                    }
                }

                crossbeam::channel::select! {
                    recv(work_rx) -> msg => {
                        if let Ok(spos) = msg {
                            match &current_visibility {
                                Some(vis) => {
                                    if vis.section_is_visible(spos) {
                                        push_job(&mut queue, &mut queued, Some(vis), spos);
                                    }
                                }
                                None => {
                                    push_job(&mut queue, &mut queued, None, spos);
                                }
                            }
                        } else {
                            break;
                        }
                    }

                    recv(visibility_rx) -> msg => {
                        if let Ok(new_vis) = msg {
                            if let Some(old_vis) = &current_visibility {
                                for dy in 0..new_vis.height {
                                    for dx in -new_vis.radius..=new_vis.radius {
                                        for dz in -new_vis.radius..=new_vis.radius {
                                            if new_vis.is_visible(dx,dy,dz) {
                                                let x = new_vis.cx + dx;
                                                let y = (new_vis.min_y / 16) + dy;
                                                let z = new_vis.cz + dz;
                                                let spos = ChunkSectionPos::new(x,y,z);

                                                if thread_dirty.lock().contains(&spos) {
                                                    push_job(&mut queue, &mut queued, Some(&new_vis), spos);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            current_visibility = Some(new_vis);
                        } else {
                            break;
                        }
                    }
                }
            }
        });

        Self {
            work_tx,
            result_rx,
            visibility_tx,
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
        let world = self.world.read();
        if let Some(chunk) = world.chunks.get(&pos) {
            let chunk = chunk.read();
            for (i, section) in chunk.sections.iter().enumerate() {
                if section.block_count > 0 {
                    let spos = ChunkSectionPos::new(pos.x, i as i32, pos.z);
                    self.submit_section(spos);
                }
            }
        }
    }

    pub fn poll(&self) -> Option<MeshResult> {
        self.result_rx.try_recv().ok()
    }

    pub fn update_visibility(&self, snapshot: VisibilitySnapshot) {
        self.visibility_tx.send(snapshot).unwrap();
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
