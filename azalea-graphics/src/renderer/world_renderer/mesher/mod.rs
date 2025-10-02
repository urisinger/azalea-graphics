use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
    io::Cursor,
    sync::Arc,
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
use parking_lot::{RwLock, Mutex};
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
    dirty: Arc<Mutex<HashSet<ChunkSectionPos>>>,
}

#[derive(Debug, Copy, Clone)]
struct Job {
    prio: f32,
    spos: ChunkSectionPos,
}

impl Eq for Job {}

impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool {
        self.spos == other.spos && self.prio == other.prio
    }
}

impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        self.prio
            .partial_cmp(&other.prio)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Job {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn prio_for(vis: &VisibilitySnapshot, spos: ChunkSectionPos) -> f32 {
    vis.section_depth(spos).unwrap_or(0.0)
}

struct SharedQueue {
    queue: Mutex<BinaryHeap<Job>>,
    queued: Mutex<HashSet<ChunkSectionPos>>,
}

impl SharedQueue {
    fn new() -> Self {
        Self {
            queue: Mutex::new(BinaryHeap::new()),
            queued: Mutex::new(HashSet::new()),
        }
    }

    fn push(&self, job: Job) {
        let mut queued = self.queued.lock();
        if queued.insert(job.spos) {
            self.queue.lock().push(job);
        }
    }

    fn pop(&self) -> Option<Job> {
        let job = self.queue.lock().pop()?;
        self.queued.lock().remove(&job.spos);
        Some(job)
    }

    fn clear_and_reprioritize(&self, vis: &VisibilitySnapshot, dirty: &HashSet<ChunkSectionPos>) {
        let mut queue = self.queue.lock();
        let mut queued = self.queued.lock();

        queue.clear();
        queued.clear();

        let side = vis.radius * 2 + 1;
        for (i, &entry) in vis.data.iter().enumerate() {
            if entry == 0.0 {
                continue;
            }

            let y = i / (side as usize * side as usize);
            let rem = i % (side as usize * side as usize);
            let z = rem / side as usize;
            let x = rem % side as usize;

            let dx = x as i32 - vis.radius;
            let dz = z as i32 - vis.radius;
            let dy = y as i32;

            let spos = ChunkSectionPos::new(vis.cx + dx, (vis.min_y / 16) + dy, vis.cz + dz);

            if dirty.contains(&spos) {
                let prio = prio_for(vis, spos);
                if queued.insert(spos) {
                    queue.push(Job { prio, spos });
                }
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.queue.lock().is_empty()
    }
}

impl Mesher {
    pub fn new(assets: Arc<Assets>, world: Arc<RwLock<azalea::world::Instance>>) -> Self {
        let num_threads = num_cpus::get().max(1);

        let (work_tx, work_rx) = unbounded::<ChunkSectionPos>();
        let (result_tx, result_rx) = unbounded::<MeshResult>();
        let (visibility_tx, visibility_rx) = unbounded::<VisibilitySnapshot>();

        let dirty = Arc::new(Mutex::new(HashSet::new()));
        let shared_queue = Arc::new(SharedQueue::new());
        let current_visibility = Arc::new(Mutex::new(None::<VisibilitySnapshot>));
        let biome_cache = Arc::new(BiomeCache::from_registries(&world.read().registries));

        // Coordinator thread
        {
            let shared_queue = Arc::clone(&shared_queue);
            let dirty = Arc::clone(&dirty);
            let current_visibility = Arc::clone(&current_visibility);

            std::thread::spawn(move || {
                loop {
                    crossbeam::channel::select! {
                        recv(work_rx) -> msg => {
                            if let Ok(spos) = msg {
                                let vis_opt = current_visibility.lock();
                                match &*vis_opt {
                                    Some(vis) if vis.section_is_visible(spos) => {
                                        let prio = prio_for(&vis, spos);
                                        shared_queue.push(Job { prio, spos });
                                    }
                                    None => {
                                        shared_queue.push(Job { prio: 0.0, spos });
                                    }
                                    _ => {}
                                }
                            } else {
                                break;
                            }
                        }

                        recv(visibility_rx) -> msg => {
                            if let Ok(new_vis) = msg {
                                let dirty_set = dirty.lock().clone();
                                shared_queue.clear_and_reprioritize(&new_vis, &dirty_set);
                                *current_visibility.lock() = Some(new_vis);
                            } else {
                                break;
                            }
                        }
                    }
                }
            });
        }

        // Worker threads
        for thread_id in 0..num_threads {
            let thread_world = Arc::clone(&world);
            let thread_dirty = Arc::clone(&dirty);
            let thread_assets = Arc::clone(&assets);
            let thread_biome_cache = Arc::clone(&biome_cache);
            let thread_queue = Arc::clone(&shared_queue);
            let thread_visibility = Arc::clone(&current_visibility);
            let thread_result_tx = result_tx.clone();

            std::thread::Builder::new()
                .name(format!("mesher-worker-{}", thread_id))
                .spawn(move || {
                    loop {
                        let job = match thread_queue.pop() {
                            Some(j) => j,
                            None => {
                                std::thread::sleep(std::time::Duration::from_millis(10));
                                continue;
                            }
                        };

                        // Check visibility again
                        if let Some(vis) = thread_visibility.lock().as_ref() {
                            if !vis.section_is_visible(job.spos) {
                                continue;
                            }
                        }

                        // Check if still dirty
                        {
                            let mut d = thread_dirty.lock();
                            if !d.remove(&job.spos) {
                                continue;
                            }
                        }

                        if let Some(local) = build_local_section(&thread_world, job.spos) {
                            let t0 = std::time::Instant::now();
                            let mesh = mesh_section(&local, &thread_biome_cache, &thread_assets);
                            let _ = thread_result_tx.send(mesh);
                            log::debug!(
                                "Thread {} meshed {:?} in {:?}",
                                thread_id,
                                job.spos,
                                t0.elapsed()
                            );
                        }
                    }
                })
                .unwrap();
        }

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
            let _ = self.work_tx.send(spos);
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
        let _ = self.visibility_tx.send(snapshot);
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
