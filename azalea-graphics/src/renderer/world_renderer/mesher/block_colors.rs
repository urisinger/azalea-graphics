use std::collections::HashMap;

use azalea::{
    block::BlockState,
    core::registry_holder::dimension_type::WorldTypeElement,
    registry::{DataRegistry, builtin::BlockKind, data::Biome},
};
use azalea_assets::Assets;
use glam::IVec3;

use crate::renderer::chunk::LocalSection;

/// Function signature for block color providers
/// Takes block_state, section, the biome registry (as a slice), local_pos,
/// tint_index, and assets
pub type BlockColorFn =
    fn(BlockState, &LocalSection, &[WorldTypeElement], IVec3, i32, &Assets) -> [f32; 3];

/// Block color registry similar to Minecraft's BlockColors
pub struct BlockColors {
    color_providers: HashMap<BlockKind, BlockColorFn>,
}

impl BlockColors {
    /// Create default block color mappings
    pub fn create_default() -> Self {
        let mut block_colors = BlockColors {
            color_providers: HashMap::new(),
        };

        block_colors.register(
            grass_color_provider,
            &[
                BlockKind::GrassBlock,
                BlockKind::Fern,
                BlockKind::ShortGrass,
                BlockKind::SugarCane,
            ],
        );

        block_colors.register(
            double_plant_grass_color_provider,
            &[BlockKind::TallGrass, BlockKind::LargeFern],
        );

        block_colors.register(
            foliage_color_provider,
            &[
                BlockKind::OakLeaves,
                BlockKind::JungleLeaves,
                BlockKind::AcaciaLeaves,
                BlockKind::DarkOakLeaves,
                BlockKind::Vine,
                BlockKind::MangroveLeaves,
            ],
        );

        block_colors.register(birch_foliage_color_provider, &[BlockKind::BirchLeaves]);
        block_colors.register(spruce_foliage_color_provider, &[BlockKind::SpruceLeaves]);
        block_colors.register(
            water_color_provider,
            &[BlockKind::Water, BlockKind::BubbleColumn],
        );
        block_colors.register(redstone_wire_color_provider, &[BlockKind::RedstoneWire]);
        block_colors.register(pumpkin_stem_color_provider, &[BlockKind::PumpkinStem]);
        block_colors.register(melon_stem_color_provider, &[BlockKind::MelonStem]);
        block_colors.register(
            attached_stem_color_provider,
            &[BlockKind::AttachedPumpkinStem, BlockKind::AttachedMelonStem],
        );
        block_colors.register(lily_pad_color_provider, &[BlockKind::LilyPad]);

        block_colors
    }

    pub fn register(&mut self, color_fn: BlockColorFn, blocks: &[BlockKind]) {
        for &block in blocks {
            self.color_providers.insert(block, color_fn);
        }
    }

    /// Get color for a block at specific tint index using direct slice indexing
    pub fn get_color(
        &self,
        block_state: BlockState,
        section: &LocalSection,
        biome_registry: &[WorldTypeElement],
        local_pos: IVec3,
        tint_index: i32,
        assets: &Assets,
    ) -> [f32; 3] {
        let block = BlockKind::from(block_state);

        if let Some(&color_fn) = self.color_providers.get(&block) {
            color_fn(
                block_state,
                section,
                biome_registry,
                local_pos,
                tint_index,
                assets,
            )
        } else {
            [1.0; 3]
        }
    }
}

// --- Provider Implementations ---

fn grass_color_provider(
    _s: BlockState,
    sec: &LocalSection,
    reg: &[WorldTypeElement],
    pos: IVec3,
    tint: i32,
    assets: &Assets,
) -> [f32; 3] {
    if tint == -1 {
        return [1.0; 3];
    }
    let biome = get_biome_at_local_pos(sec, pos);
    get_biome_grass_color(biome, reg, assets)
}

fn double_plant_grass_color_provider(
    state: BlockState,
    sec: &LocalSection,
    reg: &[WorldTypeElement],
    pos: IVec3,
    tint: i32,
    assets: &Assets,
) -> [f32; 3] {
    if tint == -1 {
        return [1.0; 3];
    }
    use azalea::block::properties::Half;
    let mut sample_pos = pos;
    if let Some(half) = state.property::<Half>() {
        if half == Half::Upper && pos.y > 0 {
            sample_pos.y -= 1;
        }
    }
    let biome = get_biome_at_local_pos(sec, sample_pos);
    get_biome_grass_color(biome, reg, assets)
}

fn foliage_color_provider(
    _s: BlockState,
    sec: &LocalSection,
    reg: &[WorldTypeElement],
    pos: IVec3,
    tint: i32,
    assets: &Assets,
) -> [f32; 3] {
    if tint == -1 {
        return [1.0; 3];
    }
    let biome = get_biome_at_local_pos(sec, pos);
    get_biome_foliage_color(biome, reg, assets)
}

fn birch_foliage_color_provider(
    _s: BlockState,
    _sec: &LocalSection,
    _reg: &[WorldTypeElement],
    _pos: IVec3,
    tint: i32,
    _assets: &Assets,
) -> [f32; 3] {
    if tint == -1 {
        return [1.0; 3];
    }
    int_color_to_rgb(-8345771)
}

fn spruce_foliage_color_provider(
    _s: BlockState,
    _sec: &LocalSection,
    _reg: &[WorldTypeElement],
    _pos: IVec3,
    tint: i32,
    _assets: &Assets,
) -> [f32; 3] {
    if tint == -1 {
        return [1.0; 3];
    }
    int_color_to_rgb(-10380959)
}

fn water_color_provider(
    _s: BlockState,
    sec: &LocalSection,
    reg: &[WorldTypeElement],
    pos: IVec3,
    tint: i32,
    _assets: &Assets,
) -> [f32; 3] {
    if tint == -1 {
        return [1.0; 3];
    }
    let biome = get_biome_at_local_pos(sec, pos);
    get_biome_water_color(biome, reg)
}

fn redstone_wire_color_provider(
    state: BlockState,
    _sec: &LocalSection,
    _reg: &[WorldTypeElement],
    _pos: IVec3,
    _tint: i32,
    _assets: &Assets,
) -> [f32; 3] {
    use azalea::block::properties::Power;
    get_color_for_power(state.property::<Power>().unwrap_or(Power::_0) as i32)
}

fn pumpkin_stem_color_provider(
    state: BlockState,
    _sec: &LocalSection,
    _reg: &[WorldTypeElement],
    _pos: IVec3,
    _tint: i32,
    _assets: &Assets,
) -> [f32; 3] {
    use azalea::block::properties::PumpkinStemAge;
    let age = state
        .property::<PumpkinStemAge>()
        .unwrap_or(PumpkinStemAge::_0) as i32;
    color(age * 32, 255 - age * 8, age * 4)
}

fn melon_stem_color_provider(
    state: BlockState,
    _sec: &LocalSection,
    _reg: &[WorldTypeElement],
    _pos: IVec3,
    _tint: i32,
    _assets: &Assets,
) -> [f32; 3] {
    use azalea::block::properties::MelonStemAge;
    let age = state.property::<MelonStemAge>().unwrap_or(MelonStemAge::_0) as i32;
    color(age * 32, 255 - age * 8, age * 4)
}

fn attached_stem_color_provider(
    _s: BlockState,
    _sec: &LocalSection,
    _reg: &[WorldTypeElement],
    _pos: IVec3,
    tint: i32,
    _assets: &Assets,
) -> [f32; 3] {
    if tint == -1 {
        return [1.0; 3];
    }
    int_color_to_rgb(-2046180)
}

fn lily_pad_color_provider(
    _s: BlockState,
    _sec: &LocalSection,
    _reg: &[WorldTypeElement],
    _pos: IVec3,
    tint: i32,
    _assets: &Assets,
) -> [f32; 3] {
    if tint == -1 {
        return [1.0; 3];
    }
    int_color_to_rgb(-14647248)
}

// --- Internal Helper Utilities ---

fn get_biome_grass_color(biome: Biome, registry: &[WorldTypeElement], assets: &Assets) -> [f32; 3] {
    if let Some(data) = registry.get(biome.protocol_id() as usize) {
        if let Some(c) = data.effects.grass_color {
            return int_color_to_rgb(c);
        }
        return get_grass_color_from_texture(
            data.temperature.clamp(0.0, 1.0) as f64,
            data.downfall.clamp(0.0, 1.0) as f64,
            assets,
        );
    }
    [1.0; 3]
}

fn get_biome_foliage_color(
    biome: Biome,
    registry: &[WorldTypeElement],
    assets: &Assets,
) -> [f32; 3] {
    if let Some(data) = registry.get(biome.protocol_id() as usize) {
        if let Some(c) = data.effects.foliage_color {
            return int_color_to_rgb(c);
        }
        return get_foliage_color_from_texture(
            data.temperature.clamp(0.0, 1.0) as f64,
            data.downfall.clamp(0.0, 1.0) as f64,
            assets,
        );
    }
    [0.2, 0.6, 0.2]
}

fn get_biome_water_color(biome: Biome, registry: &[WorldTypeElement]) -> [f32; 3] {
    registry
        .get(biome.protocol_id() as usize)
        .map(|d| int_color_to_rgb(d.effects.water_color))
        .unwrap_or([0.2, 0.4, 0.8])
}

fn get_grass_color_from_texture(t: f64, d: f64, a: &Assets) -> [f32; 3] {
    a.sample_grass_colormap(t, d).unwrap_or([1.0; 3])
}

fn get_foliage_color_from_texture(t: f64, d: f64, a: &Assets) -> [f32; 3] {
    a.sample_foliage_colormap(t, d).unwrap_or_else(|| {
        let cold = (1.0 - t) as f32;
        let dry = (1.0 - d) as f32;
        [0.1 + dry * 0.5, 0.5 + d as f32 * 0.4, 0.1 + cold * 0.2]
    })
}

fn get_biome_at_local_pos(section: &LocalSection, pos: IVec3) -> Biome {
    let x = ((pos.x - 1) / 4).clamp(0, 3) as usize;
    let y = ((pos.y - 1) / 4).clamp(0, 3) as usize;
    let z = ((pos.z - 1) / 4).clamp(0, 3) as usize;
    section.biomes[x][y][z]
}

fn int_color_to_rgb(c: i32) -> [f32; 3] {
    [
        ((c >> 16) & 0xFF) as f32 / 255.0,
        ((c >> 8) & 0xFF) as f32 / 255.0,
        (c & 0xFF) as f32 / 255.0,
    ]
}

pub fn get_color_for_power(p: i32) -> [f32; 3] {
    let r = if p == 0 {
        0.3125
    } else {
        (p as f32 / 15.0) * 0.6875 + 0.3125
    };
    [r, 0.0, 0.0]
}

pub fn color(r: i32, g: i32, b: i32) -> [f32; 3] {
    [
        r.clamp(0, 255) as f32 / 255.0,
        g.clamp(0, 255) as f32 / 255.0,
        b.clamp(0, 255) as f32 / 255.0,
    ]
}
