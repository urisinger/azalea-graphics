use azalea::{core::direction::Direction, ecs::{entity::Entity, world::World}};

use crate::renderer::entity_renderer::{EntityPose, state::entity::EntityRenderState};

#[derive(Debug, Clone)]
pub struct LivingEntityRenderState {
    pub parent: EntityRenderState,
    pub body_yaw: f32,
    pub relative_head_yaw: f32,
    pub pitch: f32,
    pub death_time: f32,
    pub limb_swing_animation_progress: f32,
    pub limb_swing_amplitude: f32,
    pub base_scale: f32,
    pub age_scale: f32,
    pub flip_upside_down: bool,
    pub shaking: bool,
    pub baby: bool,
    pub touching_water: bool,
    pub using_riptide: bool,
    pub hurt: bool,
    pub invisible_to_player: bool,
    pub sleeping_direction: Option<Direction>,
    pub pose: EntityPose,
    //pub head_item_render_state: ItemRenderState,
    pub head_item_animation_progress: f32,
    //pub wearing_skull_type: Option<SkullType>,
    //pub wearing_skull_profile: Option<SkullProfile>,
}

impl LivingEntityRenderState {
    pub fn new(world: &mut World, entity: Entity) -> Self {
        Self {
            parent: EntityRenderState::new(world, entity),
            body_yaw: 0.0,
            relative_head_yaw: 0.0,
            pitch: 0.0,
            death_time: 0.0,
            limb_swing_animation_progress: 0.0,
            limb_swing_amplitude: 0.0,
            base_scale: 1.0, // matches Java default
            age_scale: 1.0,  // matches Java default
            flip_upside_down: false,
            shaking: false,
            baby: false,
            touching_water: false,
            using_riptide: false,
            hurt: false,
            invisible_to_player: false,
            sleeping_direction: None,
            pose: EntityPose::Standing, // matches Java default
            head_item_animation_progress: 0.0,
        }
    }
}
