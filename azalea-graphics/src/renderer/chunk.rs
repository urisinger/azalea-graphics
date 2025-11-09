use std::sync::Arc;

use azalea::{
    blocks::BlockState,
    core::position::{ChunkSectionBiomePos, ChunkSectionBlockPos, ChunkSectionPos},
    registry::Biome,
    world::Chunk,
};
use parking_lot::RwLock;

pub struct LocalSection {
    pub blocks: Box<[[[Option<BlockState>; 18]; 18]; 18]>,
    pub biomes: Box<[[[Biome; 4]; 4]; 4]>,
    pub spos: ChunkSectionPos,
}

const NORTH: usize = 0;
const SOUTH: usize = 1;
const EAST: usize = 2;
const WEST: usize = 3;
const NE: usize = 4;
const NW: usize = 5;
const SE: usize = 6;
const SW: usize = 7;

pub struct LocalChunk {
    pub center: Arc<RwLock<Chunk>>,
    pub neighbors: [Option<Arc<RwLock<Chunk>>>; 8],

    pub min_y: i32,
}

pub struct BorrowedChunks<'a> {
    pub center: parking_lot::RwLockReadGuard<'a, Chunk>,
    pub neighbors: [Option<parking_lot::RwLockReadGuard<'a, Chunk>>; 8],

    pub min_y: i32,
}

impl LocalChunk {
    pub fn borrow_chunks(&self) -> BorrowedChunks<'_> {
        let center = self.center.read();
        let neighbors = [
            self.neighbors[NORTH].as_ref().map(|c| c.read()),
            self.neighbors[SOUTH].as_ref().map(|c| c.read()),
            self.neighbors[EAST].as_ref().map(|c| c.read()),
            self.neighbors[WEST].as_ref().map(|c| c.read()),
            self.neighbors[NE].as_ref().map(|c| c.read()),
            self.neighbors[NW].as_ref().map(|c| c.read()),
            self.neighbors[SE].as_ref().map(|c| c.read()),
            self.neighbors[SW].as_ref().map(|c| c.read()),
        ];

        BorrowedChunks {
            center,
            neighbors,
            min_y: self.min_y,
        }
    }
}

impl<'a> BorrowedChunks<'a> {
    /// Build a single local section with 18x18x18 extended block data
    pub fn build_local_section(&self, spos: ChunkSectionPos) -> LocalSection {
        let mut blocks = Box::new([[[None; 18]; 18]; 18]);
        let mut biomes = Box::new([[[Default::default(); 4]; 4]; 4]);

        for lx in -1..17 {
            for ly in -1..17 {
                for lz in -1..17 {
                    let ix = (lx + 1) as usize;
                    let iy = (ly + 1) as usize;
                    let iz = (lz + 1) as usize;

                    blocks[ix][iy][iz] = self.get_block_local(spos.y - self.min_y, lx, ly, lz);
                }
            }
        }

        // Copy biome data from the center chunk section
        if let Some(section) = self.center.sections.get((spos.y - self.min_y) as usize) {
            for x in 0..4 {
                for y in 0..4 {
                    for z in 0..4 {
                        let pos = ChunkSectionBiomePos { x, y, z };
                        biomes[x as usize][y as usize][z as usize] = section.get_biome(pos);
                    }
                }
            }
        }

        LocalSection {
            blocks,
            biomes,
            spos,
        }
    }

    pub fn get_block_local(&self, base_y: i32, lx: i32, ly: i32, lz: i32) -> Option<BlockState> {
        let cx_off = lx.div_euclid(16);
        let sx = lx.rem_euclid(16) as u8;

        let cy_off = ly.div_euclid(16);
        let sy = ly.rem_euclid(16) as u8;

        let cz_off = lz.div_euclid(16);
        let sz = lz.rem_euclid(16) as u8;

        let chunk_ref = match (cx_off, cz_off) {
            (0, 0) => Some(&*self.center),
            (0, -1) => self.neighbors[NORTH].as_deref(),
            (0, 1) => self.neighbors[SOUTH].as_deref(),
            (-1, 0) => self.neighbors[WEST].as_deref(),
            (1, 0) => self.neighbors[EAST].as_deref(),
            (-1, -1) => self.neighbors[NW].as_deref(),
            (1, -1) => self.neighbors[NE].as_deref(),
            (-1, 1) => self.neighbors[SW].as_deref(),
            (1, 1) => self.neighbors[SE].as_deref(),
            _ => None,
        };

        if let Some(chunk) = chunk_ref {
            let section_index = (base_y + cy_off) as usize;
            if let Some(section) = chunk.sections.get(section_index) {
                return Some(section.get_block_state(ChunkSectionBlockPos {
                    x: sx,
                    y: sy,
                    z: sz,
                }));
            }
        }
        None
    }
}
