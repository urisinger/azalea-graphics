use azalea::{
    core::arm::Arm,
    ecs::{entity::Entity, world::World},
    inventory::ItemStack,
};

use crate::renderer::entity_renderer::state::armed_entity::ArmedEntityRenderState;

#[derive(Debug, Clone)]
pub struct BipedRenderState {
    pub parent: ArmedEntityRenderState,
    pub leaning_pitch: f32,
    pub hand_swing_progress: f32,
    pub limb_amplitude_inverse: f32,
    pub crossbow_pull_time: f32,
    pub item_use_time: i32,
    pub preferred_arm: Arm,
    // pub active_hand: Hand,
    pub is_in_sneaking_pose: bool,
    pub is_gliding: bool,
    pub is_swimming: bool,
    pub has_vehicle: bool,
    pub is_using_item: bool,
    pub left_wing_pitch: f32,
    pub left_wing_yaw: f32,
    pub left_wing_roll: f32,
    pub equipped_head_stack: ItemStack,
    pub equipped_chest_stack: ItemStack,
    pub equipped_legs_stack: ItemStack,
    pub equipped_feet_stack: ItemStack,
}

impl BipedRenderState {
    pub fn new(world: &mut World, entity: Entity) -> Self {
        Self {
            parent: ArmedEntityRenderState::new(world, entity),
            limb_amplitude_inverse: 1.0,
            equipped_head_stack: ItemStack::Empty,
            equipped_chest_stack: ItemStack::Empty,
            equipped_legs_stack: ItemStack::Empty,
            equipped_feet_stack: ItemStack::Empty,
            preferred_arm: Arm::Right,
            leaning_pitch: 0.0,
            hand_swing_progress: 0.0,
            crossbow_pull_time: 0.0,
            item_use_time: 0,
            is_in_sneaking_pose: false,
            is_gliding: false,
            is_swimming: false,
            has_vehicle: false,
            is_using_item: false,
            left_wing_pitch: 0.0,
            left_wing_yaw: 0.0,
            left_wing_roll: 0.0,
        }
    }
}
