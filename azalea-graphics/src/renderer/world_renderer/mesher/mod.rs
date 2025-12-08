use std::{
    cmp::Ordering,
    collections::HashSet,
    io::Cursor,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
    },
};

use azalea::{
    blocks::BlockState,
    core::{
        position::{ChunkPos, ChunkSectionPos},
        registry_holder::{BiomeData, RegistryHolder},
    },
    registry::{Biome, Block, DataRegistry},
};
use azalea_assets::Assets;
use crossbeam::channel::{Receiver, Sender, unbounded};
use glam::IVec3;
use log::error;
use parking_lot::{Mutex, RwLock};
use simdnbt::Deserialize;

use crate::renderer::{
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

struct WorkerContext {
    world: Arc<RwLock<azalea::world::Instance>>,
    dirty: Arc<Mutex<HashSet<ChunkSectionPos>>>,
    assets: Arc<Assets>,
    biome_cache: BiomeCache,
    shared_queue: SharedQueue,
    current_visibility: Mutex<Option<VisibilitySnapshot>>,
    result_tx: Sender<MeshResult>,
    should_stop: AtomicBool,

    total_mesh_time_ns: AtomicU64,
    total_meshes: AtomicU64,
}

pub struct Mesher {
    result_rx: Receiver<MeshResult>,
    visibility_tx: Sender<VisibilitySnapshot>,

    pub world: Arc<RwLock<azalea::world::Instance>>,
    dirty: Arc<Mutex<HashSet<ChunkSectionPos>>>,
    assets: Arc<Assets>,

    worker_ctx: Arc<WorkerContext>,

    worker_count: u32,

    total_mesh_time_ns: AtomicU64,
    total_meshes: AtomicU64,
}

#[derive(Debug, Copy, Clone)]
struct Job {
    prio: f32,
    spos: ChunkSectionPos,
}

fn prio_for(vis: &VisibilitySnapshot, spos: ChunkSectionPos) -> f32 {
    vis.section_depth(spos).unwrap_or(0.0)
}

struct SharedQueue {
    jobs: RwLock<Arc<Vec<Job>>>,
    next_job_index: AtomicUsize,
    parked_threads: Mutex<Vec<std::thread::Thread>>,
}

impl SharedQueue {
    fn new() -> Self {
        Self {
            jobs: RwLock::new(Arc::new(Vec::new())),
            next_job_index: AtomicUsize::new(0),
            parked_threads: Mutex::new(Vec::new()),
        }
    }

    fn pop(&self, should_stop: &AtomicBool) -> Option<Job> {
        loop {
            if should_stop.load(AtomicOrdering::Acquire) {
                return None;
            }

            let idx = self.next_job_index.fetch_add(1, AtomicOrdering::Relaxed);

            let jobs = self.jobs.read();

            if idx < jobs.len() {
                return Some(jobs[idx]);
            }

            drop(jobs);

            let idx = self.next_job_index.load(AtomicOrdering::Relaxed);
            let jobs = self.jobs.read();
            if idx < jobs.len() {
                drop(jobs);
                continue;
            }
            drop(jobs);

            if should_stop.load(AtomicOrdering::Acquire) {
                return None;
            }

            self.parked_threads.lock().push(std::thread::current());
            std::thread::park();
        }
    }

    fn clear_and_reprioritize(&self, vis: &VisibilitySnapshot, dirty: &HashSet<ChunkSectionPos>) {
        let mut jobs = Vec::new();
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

            let spos = ChunkSectionPos::new(vis.cx + dx, vis.min_y + dy, vis.cz + dz);

            if dirty.contains(&spos) {
                let prio = prio_for(vis, spos);
                jobs.push(Job { prio, spos });
            }
        }

        jobs.sort_unstable_by(|a, b| b.prio.partial_cmp(&a.prio).unwrap_or(Ordering::Equal));

        let mut guard = self.jobs.write();
        *guard = Arc::new(jobs);
        self.next_job_index.store(0, AtomicOrdering::Release);
        drop(guard);

        let mut parked = self.parked_threads.lock();
        for thread in parked.drain(..) {
            thread.unpark();
        }
    }

    fn is_empty(&self) -> bool {
        let idx = self.next_job_index.load(AtomicOrdering::Relaxed);
        let jobs = self.jobs.read();
        idx >= jobs.len()
    }
}

impl Mesher {
    pub fn new(assets: Arc<Assets>, world: Arc<RwLock<azalea::world::Instance>>) -> Self {
        let num_threads = num_cpus::get().max(1) as u32 / 2;

        let (result_tx, result_rx) = unbounded::<MeshResult>();
        let (visibility_tx, visibility_rx) = unbounded::<VisibilitySnapshot>();

        let dirty = Arc::new(Mutex::new(HashSet::new()));
        let shared_queue = SharedQueue::new();
        let current_visibility = Mutex::new(None::<VisibilitySnapshot>);
        let biome_cache = BiomeCache::from_registries(&world.read().registries);
        let should_stop = AtomicBool::new(false);

        let worker_ctx = Arc::new(WorkerContext {
            world: Arc::clone(&world),
            dirty: Arc::clone(&dirty),
            assets: Arc::clone(&assets),
            biome_cache,
            shared_queue,
            current_visibility,
            result_tx,
            should_stop,
            total_mesh_time_ns: AtomicU64::new(0),
            total_meshes: AtomicU64::new(0),
        });

        {
            let ctx = Arc::clone(&worker_ctx);
            std::thread::spawn(move || {
                loop {
                    match visibility_rx.recv() {
                        Ok(new_vis) => {
                            let dirty_set = ctx.dirty.lock().clone();
                            ctx.shared_queue
                                .clear_and_reprioritize(&new_vis, &dirty_set);
                            *ctx.current_visibility.lock() = Some(new_vis);
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        for i in 0..num_threads {
            Self::spawn_worker(i, Arc::clone(&worker_ctx));
        }

        Self {
            result_rx,
            visibility_tx,
            world,
            dirty,
            assets,
            worker_ctx,
            worker_count: num_threads,

            total_mesh_time_ns: AtomicU64::new(0),
            total_meshes: AtomicU64::new(0),
        }
    }

    pub fn average_mesh_time_ns(&self) -> f32 {
        let count = self.worker_ctx.total_meshes.load(AtomicOrdering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let total_ns = self
            .worker_ctx
            .total_mesh_time_ns
            .load(AtomicOrdering::Relaxed);
        total_ns as f32 / count as f32
    }

    pub fn average_mesh_time_ms(&self) -> f32 {
        self.average_mesh_time_ns() / 1_000_000.0
    }

    pub fn submit_section(&self, spos: ChunkSectionPos) {
        self.dirty.lock().insert(spos);
    }

    pub fn submit_chunk(&self, pos: ChunkPos) {
        let world = self.world.read();
        let min = world.chunks.min_y / 16;
        let max = min + world.chunks.height as i32 / 16;
        for y in min..max {
            let spos = ChunkSectionPos::new(pos.x, y, pos.z);
            self.submit_section(spos);
        }
    }

    pub fn poll(&self) -> Option<MeshResult> {
        self.result_rx.try_recv().ok()
    }

    pub fn update_visibility(&self, snapshot: VisibilitySnapshot) {
        let _ = self.visibility_tx.send(snapshot);
    }

    fn spawn_worker(id: u32, ctx: Arc<WorkerContext>) {
        std::thread::Builder::new()
            .name(format!("mesher-worker-{}", id))
            .spawn(move || {
                loop {
                    let job = match ctx.shared_queue.pop(&ctx.should_stop) {
                        Some(j) => j,
                        None => break,
                    };

                    {
                        let mut d = ctx.dirty.lock();
                        if !d.remove(&job.spos) {
                            continue;
                        }
                    }

                    if let Some(local) = build_local_section(&ctx.world, job.spos) {
                        let t0 = std::time::Instant::now();
                        let mesh = mesh_section(&local, &ctx.biome_cache, &ctx.assets);
                        let elapsed = t0.elapsed();

                        let nanos = elapsed.as_nanos() as u64;

                        ctx.total_mesh_time_ns
                            .fetch_add(nanos, AtomicOrdering::Relaxed);
                        ctx.total_meshes.fetch_add(1, AtomicOrdering::Relaxed);

                        let _ = ctx.result_tx.send(mesh);
                    }
                }
            })
            .unwrap();
    }

    pub fn set_worker_threads(&mut self, new_thread_count: u32) {
        let current = self.worker_count;

        if new_thread_count == current {
            return;
        }

        if new_thread_count > current {
            for i in current..new_thread_count {
                Self::spawn_worker(i, Arc::clone(&self.worker_ctx));
            }
        }

        self.worker_count = new_thread_count;
        log::info!(
            "Worker thread count changed from {} to {}",
            current,
            new_thread_count
        );
    }

    pub fn get_worker_thread_count(&self) -> u32 {
        self.worker_count
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

    let local_chunk = LocalChunk {
        center,
        neighbors,
        min_y: world_guard.chunks.min_y / 16,
    };
    drop(world_guard);

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
            .get(&azalea::Identifier::new(Biome::NAME))
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
