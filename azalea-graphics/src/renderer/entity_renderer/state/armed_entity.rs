use azalea::{core::arm::Arm, ecs::{entity::Entity, world::World}};

use crate::renderer::entity_renderer::{ArmPose, state::living_entity::LivingEntityRenderState};

#[derive(Debug, Clone)]
pub struct ArmedEntityRenderState {
    pub parent: LivingEntityRenderState,
    pub main_arm: Arm,
    pub right_arm_pose: ArmPose,
    //right_hand_item_state: ItemRenderState,
    pub left_arm_pose: ArmPose,
    //left_hand_item_state: ItemRenderState,
}

impl ArmedEntityRenderState {
    pub fn new(world: &mut World, entity: Entity) -> Self {
        Self {
            parent: LivingEntityRenderState::new(world, entity),
            main_arm: Arm::Right,
            left_arm_pose: ArmPose::Empty,
            right_arm_pose: ArmPose::Empty,
        }
    }
}
