use std::f32::consts::PI;

use azalea_assets::processed::entity_model::{Model, Transform};

use crate::renderer::entity_renderer::{
    models::biped::HumanoidModel, state::zombie::ZombieRenderState, transform::ModelTransforms,
};

pub struct ZombieModel<'a> {
    biped: HumanoidModel<'a>,
}

impl<'a> ZombieModel<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self {
            biped: HumanoidModel::new(model),
        }
    }

    pub fn set_angles(&self, transforms: &mut ModelTransforms, state: &ZombieRenderState) {
        // Call parent biped animation first
        //self.biped.set_angles(transforms, &state.parent);

        // Apply zombie-specific arm animations
        let swing_progress = state.hand_swing_progress;
        let attacking = state.attacking;
        let age = state.age;

        //self.zombie_arms(transforms, attacking, swing_progress, age);
    }

    fn zombie_arms(
        &self,
        transforms: &mut ModelTransforms,
        attacking: bool,
        swing_progress: f32,
        animation_progress: f32,
    ) {
        let f = (swing_progress * PI).sin();
        let g = ((1.0 - (1.0 - swing_progress) * (1.0 - swing_progress)) * PI).sin();

        let h = -PI / if attacking { 1.5 } else { 2.25 };
        // Get references to arms
        let right_arm = transforms.get_mut(self.biped.right_arm);

        right_arm.rotation.z = 0.0;
        right_arm.rotation.y = -(0.1 - f * 0.6);
        right_arm.rotation.x = h;
        right_arm.rotation.x += f * 1.2 - g * 0.4;
        Self::swing_arm(right_arm, animation_progress, 1.0);

        let left_arm = transforms.get_mut(self.biped.left_arm);
        left_arm.rotation.z = 0.0;
        left_arm.rotation.y = 0.1 - f * 0.6;
        left_arm.rotation.x = h;
        left_arm.rotation.x += f * 1.2 - g * 0.4;

        Self::swing_arm(left_arm, animation_progress, -1.0);
    }

    fn swing_arm(arm_transform: &mut Transform, animation_progress: f32, sigma: f32) {
        arm_transform.rotation.z += sigma * ((animation_progress * 0.09).cos() * 0.05 + 0.05);
        arm_transform.rotation.x += sigma * (animation_progress * 0.067).sin() * 0.05;
    }
}
