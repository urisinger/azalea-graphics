use std::f32::consts::PI;

use azalea_assets::processed::entity_model::{Model, ModelPart};

use crate::renderer::entity_renderer::{
    ArmPose, state::humanoid::BipedRenderState, transform::ModelTransforms,
};

pub struct HumanoidModel<'a> {
    pub head: &'a ModelPart,
    pub hat: &'a ModelPart,
    pub body: &'a ModelPart,
    pub right_arm: &'a ModelPart,
    pub left_arm: &'a ModelPart,
    pub right_leg: &'a ModelPart,
    pub left_leg: &'a ModelPart,
}

impl<'a> HumanoidModel<'a> {
    pub fn new(model: &'a Model) -> Self {
        let root = &model.part;

        let head = root
            .children
            .get("head")
            .expect("Biped model missing 'head' part");
        let hat = head
            .children
            .get("hat")
            .expect("Biped model missing 'hat' part");
        let body = root
            .children
            .get("body")
            .expect("Biped model missing 'body' part");
        let right_arm = root
            .children
            .get("right_arm")
            .expect("Biped model missing 'right_arm' part");
        let left_arm = root
            .children
            .get("left_arm")
            .expect("Biped model missing 'left_arm' part");
        let right_leg = root
            .children
            .get("right_leg")
            .expect("Biped model missing 'right_leg' part");
        let left_leg = root
            .children
            .get("left_leg")
            .expect("Biped model missing 'left_leg' part");

        Self {
            head,
            hat,
            body,
            right_arm,
            left_arm,
            right_leg,
            left_leg,
        }
    }

    pub fn set_angles(&self, transforms: &mut ModelTransforms, state: &BipedRenderState) {
        let left_arm_pose = &state.left_arm_pose;
        let right_arm_pose = &state.right_arm_pose;
        let leaning_pitch = state.leaning_pitch;
        let is_gliding = state.is_gliding;

        // Head angles (Deref makes parent fields accessible!)
        let head = transforms.get_mut(self.head);
        head.rotation.x = state.pitch * (PI / 180.0);
        head.rotation.y = state.relative_head_yaw * (PI / 180.0);

        if is_gliding {
            head.rotation.x = -PI / 4.0;
        } else if leaning_pitch > 0.0 {
            head.rotation.x = lerp_angle(leaning_pitch, head.rotation.x, -PI / 4.0);
        }

        // Limb animation
        let limb_swing = state.limb_swing_animation_progress;
        let limb_amplitude = state.limb_swing_amplitude;
        let limb_inverse = state.limb_amplitude_inverse;

        let right_arm = transforms.get_mut(self.right_arm);
        right_arm.rotation.x =
            (limb_swing * 0.6662 + PI).cos() * 2.0 * limb_amplitude * 0.5 / limb_inverse;

        let left_arm = transforms.get_mut(self.left_arm);
        left_arm.rotation.x =
            (limb_swing * 0.6662).cos() * 2.0 * limb_amplitude * 0.5 / limb_inverse;

        let right_leg = transforms.get_mut(self.right_leg);
        right_leg.rotation.x = (limb_swing * 0.6662).cos() * 1.4 * limb_amplitude / limb_inverse;
        right_leg.rotation.y = 0.005;
        right_leg.rotation.z = 0.005;

        let left_leg = transforms.get_mut(self.left_leg);
        left_leg.rotation.x =
            (limb_swing * 0.6662 + PI).cos() * 1.4 * limb_amplitude / limb_inverse;
        left_leg.rotation.y = -0.005;
        left_leg.rotation.z = -0.005;

        // Vehicle adjustments
        if state.has_vehicle {
            let right_arm = transforms.get_mut(self.right_arm);
            right_arm.rotation.x += -PI / 5.0;

            let left_arm = transforms.get_mut(self.left_arm);
            left_arm.rotation.x += -PI / 5.0;

            let right_leg = transforms.get_mut(self.right_leg);
            right_leg.rotation.x = -1.4137167;
            right_leg.rotation.y = PI / 10.0;
            right_leg.rotation.z = 0.07853982;

            let left_leg = transforms.get_mut(self.left_leg);
            left_leg.rotation.x = -1.4137167;
            left_leg.rotation.y = -PI / 10.0;
            left_leg.rotation.z = -0.07853982;
        }

        // Arm poses
        let main_arm_right = state.main_arm == azalea::core::arm::Arm::Right;
        if state.is_using_item {
            // Handle item usage - simplified for now
            self.position_right_arm(transforms, state, right_arm_pose);
            self.position_left_arm(transforms, state, left_arm_pose);
        } else {
            let two_handed = if main_arm_right {
                left_arm_pose.is_two_handed()
            } else {
                right_arm_pose.is_two_handed()
            };

            if main_arm_right != two_handed {
                self.position_left_arm(transforms, state, left_arm_pose);
                self.position_right_arm(transforms, state, right_arm_pose);
            } else {
                self.position_right_arm(transforms, state, right_arm_pose);
                self.position_left_arm(transforms, state, left_arm_pose);
            }
        }

        // Arm swinging animation
        self.animate_arms(transforms, state);

        // Sneaking pose
        if state.is_in_sneaking_pose {
            let body = transforms.get_mut(self.body);
            body.rotation.x = 0.5;

            let right_arm = transforms.get_mut(self.right_arm);
            right_arm.rotation.x += 0.4;

            let left_arm = transforms.get_mut(self.left_arm);
            left_arm.rotation.x += 0.4;

            let right_leg = transforms.get_mut(self.right_leg);
            right_leg.pivot.z += 4.0;

            let left_leg = transforms.get_mut(self.left_leg);
            left_leg.pivot.z += 4.0;

            let head = transforms.get_mut(self.head);
            head.pivot.y += 4.2;

            let body = transforms.get_mut(self.body);
            body.pivot.y += 3.2;

            let left_arm = transforms.get_mut(self.left_arm);
            left_arm.pivot.y += 3.2;

            let right_arm = transforms.get_mut(self.right_arm);
            right_arm.pivot.y += 3.2;
        }

        // Swimming animation
        if leaning_pitch > 0.0 {
            let limb_pos = limb_swing % 26.0;

            if !state.is_using_item {
                if limb_pos < 14.0 {
                    let curve_value = self.swimming_curve(limb_pos) / self.swimming_curve(14.0);

                    let left_arm = transforms.get_mut(self.left_arm);
                    left_arm.rotation.x = lerp_angle(leaning_pitch, left_arm.rotation.x, 0.0);
                    left_arm.rotation.y = lerp_angle(leaning_pitch, left_arm.rotation.y, PI);
                    left_arm.rotation.z = lerp_angle(
                        leaning_pitch,
                        left_arm.rotation.z,
                        PI + 1.8707964 * curve_value,
                    );

                    let right_arm = transforms.get_mut(self.right_arm);
                    right_arm.rotation.x = lerp(leaning_pitch, right_arm.rotation.x, 0.0);
                    right_arm.rotation.y = lerp(leaning_pitch, right_arm.rotation.y, PI);
                    right_arm.rotation.z = lerp(
                        leaning_pitch,
                        right_arm.rotation.z,
                        PI - 1.8707964 * curve_value,
                    );
                }
            }

            let left_leg = transforms.get_mut(self.left_leg);
            left_leg.rotation.x = lerp(
                leaning_pitch,
                left_leg.rotation.x,
                0.3 * (limb_swing * 0.33333334 + PI).cos(),
            );

            let right_leg = transforms.get_mut(self.right_leg);
            right_leg.rotation.x = lerp(
                leaning_pitch,
                right_leg.rotation.x,
                0.3 * (limb_swing * 0.33333334).cos(),
            );
        }
    }

    fn position_right_arm(
        &self,
        transforms: &mut ModelTransforms,
        state: &BipedRenderState,
        arm_pose: &ArmPose,
    ) {
        match arm_pose {
            ArmPose::Empty => {
                transforms.get_mut(self.right_arm).rotation.y = 0.0;
            }
            ArmPose::Item => {
                let right_arm = transforms.get_mut(self.right_arm);
                right_arm.rotation.x = right_arm.rotation.x * 0.5 - PI / 10.0;
                right_arm.rotation.y = 0.0;
            }
            ArmPose::Block => {
                self.position_blocking_arm(transforms, self.right_arm, true, state);
            }
            ArmPose::BowAndArrow => {
                // Read head values first
                let head_rot = transforms.get(self.head).rotation;
                let right_arm = transforms.get_mut(self.right_arm);
                right_arm.rotation.y = -0.1 + head_rot.y;
                right_arm.rotation.x = -PI / 2.0 + head_rot.x;
            }
            ArmPose::ThrowSpear => {
                let right_arm = transforms.get_mut(self.right_arm);
                right_arm.rotation.x = right_arm.rotation.x * 0.5 - PI;
                right_arm.rotation.y = 0.0;
            }
            ArmPose::Spyglass => {
                let is_sneaking = if state.is_in_sneaking_pose {
                    0.2617994
                } else {
                    0.0
                };
                // Read head values first
                let head_rot = transforms.get(self.head).rotation;
                let right_arm = transforms.get_mut(self.right_arm);
                right_arm.rotation.x = (head_rot.x - 1.9198622 - is_sneaking).clamp(-2.4, 3.3);
                right_arm.rotation.y = head_rot.y - 0.2617994;
            }
            ArmPose::TootHorn => {
                // Read head values first
                let head_rot = transforms.get(self.head).rotation;
                let right_arm = transforms.get_mut(self.right_arm);
                right_arm.rotation.x = head_rot.x.clamp(-1.2, 1.2) - 1.4835298;
                right_arm.rotation.y = head_rot.y - PI / 6.0;
            }
            ArmPose::Brush => {
                let right_arm = transforms.get_mut(self.right_arm);
                right_arm.rotation.x = right_arm.rotation.x * 0.5 - PI / 5.0;
                right_arm.rotation.y = 0.0;
            }
            _ => {}
        }
    }

    fn position_left_arm(
        &self,
        transforms: &mut ModelTransforms,
        state: &BipedRenderState,
        arm_pose: &ArmPose,
    ) {
        match arm_pose {
            ArmPose::Empty => {
                transforms.get_mut(self.left_arm).rotation.y = 0.0;
            }
            ArmPose::Item => {
                let left_arm = transforms.get_mut(self.left_arm);
                left_arm.rotation.x = left_arm.rotation.x * 0.5 - PI / 10.0;
                left_arm.rotation.y = 0.0;
            }
            ArmPose::Block => {
                self.position_blocking_arm(transforms, self.left_arm, false, state);
            }
            ArmPose::BowAndArrow => {
                // Read head values first
                let head_rot = transforms.get(self.head).rotation;
                let left_arm = transforms.get_mut(self.left_arm);
                left_arm.rotation.y = 0.1 + head_rot.y;
                left_arm.rotation.x = -PI / 2.0 + head_rot.x;
            }
            ArmPose::ThrowSpear => {
                let left_arm = transforms.get_mut(self.left_arm);
                left_arm.rotation.x = left_arm.rotation.x * 0.5 - PI;
                left_arm.rotation.y = 0.0;
            }
            ArmPose::Spyglass => {
                let is_sneaking = if state.is_in_sneaking_pose {
                    0.2617994
                } else {
                    0.0
                };
                // Read head values first
                let head_rot = transforms.get(self.head).rotation;
                let left_arm = transforms.get_mut(self.left_arm);
                left_arm.rotation.x = (head_rot.x - 1.9198622 - is_sneaking).clamp(-2.4, 3.3);
                left_arm.rotation.y = head_rot.y + 0.2617994;
            }
            ArmPose::TootHorn => {
                // Read head values first
                let head_rot = transforms.get(self.head).rotation;
                let left_arm = transforms.get_mut(self.left_arm);
                left_arm.rotation.x = head_rot.x.clamp(-1.2, 1.2) - 1.4835298;
                left_arm.rotation.y = head_rot.y + PI / 6.0;
            }
            ArmPose::Brush => {
                let left_arm = transforms.get_mut(self.left_arm);
                left_arm.rotation.x = left_arm.rotation.x * 0.5 - PI / 5.0;
                left_arm.rotation.y = 0.0;
            }
            _ => {}
        }
    }

    fn position_blocking_arm(
        &self,
        transforms: &mut ModelTransforms,
        arm: &ModelPart,
        is_right: bool,
        state: &BipedRenderState,
    ) {
        // Read head rotation first
        let head_rot = transforms.get(self.head).rotation;
        let arm_transform = transforms.get_mut(arm);

        arm_transform.rotation.x =
            arm_transform.rotation.x * 0.5 - 0.9424779 + head_rot.x.clamp(-1.3962634, 0.43633232);
        arm_transform.rotation.y = (if is_right { -30.0 } else { 30.0 }) * (PI / 180.0)
            + head_rot.y.clamp(-PI / 6.0, PI / 6.0);
    }

    fn animate_arms(&self, transforms: &mut ModelTransforms, state: &BipedRenderState) {
        let swing_progress = state.hand_swing_progress;
        if swing_progress <= 0.0 {
            return;
        }

        let body = transforms.get_mut(self.body);
        body.rotation.y = (swing_progress.sqrt() * PI * 2.0).sin() * 0.2;

        if state.preferred_arm == azalea::core::arm::Arm::Left {
            body.rotation.y *= -1.0;
        }

        let body_yaw = transforms.get(self.body).rotation.y;
        let age_scale = state.age_scale;

        let right_arm = transforms.get_mut(self.right_arm);
        right_arm.pivot.z = body_yaw.sin() * 5.0 * age_scale;
        right_arm.pivot.x = -body_yaw.cos() * 5.0 * age_scale;
        right_arm.rotation.y += body_yaw;

        let left_arm = transforms.get_mut(self.left_arm);
        left_arm.pivot.z = -body_yaw.sin() * 5.0 * age_scale;
        left_arm.pivot.x = body_yaw.cos() * 5.0 * age_scale;
        left_arm.rotation.y += body_yaw;
        left_arm.rotation.x += body_yaw;

        let swinging_arm = if state.preferred_arm == azalea::core::arm::Arm::Right {
            self.right_arm
        } else {
            self.left_arm
        };

        let mut g = 1.0 - swing_progress;
        g *= g;
        g *= g;
        g = 1.0 - g;

        let i = (g * PI).sin();
        let head_pitch = transforms.get(self.head).rotation.x;
        let j = (swing_progress * PI).sin() * -(head_pitch - 0.7) * 0.75;

        let arm = transforms.get_mut(swinging_arm);
        arm.rotation.x -= i * 1.2 + j;
        arm.rotation.y += body_yaw * 2.0;
        arm.rotation.z += (swing_progress * PI).sin() * -0.4;
    }

    fn swimming_curve(&self, value: f32) -> f32 {
        -65.0 * value + value * value
    }
}

impl ArmPose {
    pub fn is_two_handed(&self) -> bool {
        matches!(self, ArmPose::BowAndArrow | ArmPose::CrossbowHold)
    }
}

// Helper functions
fn lerp(delta: f32, start: f32, end: f32) -> f32 {
    start + delta * (end - start)
}

fn lerp_angle(delta: f32, start: f32, end: f32) -> f32 {
    start + delta * wrap_degrees(end - start)
}

fn wrap_degrees(degrees: f32) -> f32 {
    let mut value = degrees % (2.0 * PI);
    if value >= PI {
        value -= 2.0 * PI;
    }
    if value < -PI {
        value += 2.0 * PI;
    }
    value
}
