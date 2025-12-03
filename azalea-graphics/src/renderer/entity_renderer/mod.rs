use std::sync::Arc;

use azalea_assets::Assets;

use crate::renderer::entity_renderer::state::RenderState;


mod renderers;
mod state;

pub struct EntityRenderer {
    assets: Arc<Assets>,
}

impl EntityRenderer {
    pub fn new(assets: Arc<Assets>) -> Self {
        Self { assets }
    }

    pub fn render(&mut self, states: Vec<RenderState>){

    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityPose {
    Standing = 0,
    Gliding = 1,
    Sleeping = 2,
    Swimming = 3,
    SpinAttack = 4,
    Crouching = 5,
    LongJumping = 6,
    Dying = 7,
    Croaking = 8,
    UsingTongue = 9,
    Sitting = 10,
    Roaring = 11,
    Sniffing = 12,
    Emerging = 13,
    Digging = 14,
    Sliding = 15,
    Shooting = 16,
    Inhaling = 17,
}

impl EntityPose {
    pub fn index(self) -> u8 {
        self as u8
    }

    pub fn from_index(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Standing),
            1 => Some(Self::Gliding),
            2 => Some(Self::Sleeping),
            3 => Some(Self::Swimming),
            4 => Some(Self::SpinAttack),
            5 => Some(Self::Crouching),
            6 => Some(Self::LongJumping),
            7 => Some(Self::Dying),
            8 => Some(Self::Croaking),
            9 => Some(Self::UsingTongue),
            10 => Some(Self::Sitting),
            11 => Some(Self::Roaring),
            12 => Some(Self::Sniffing),
            13 => Some(Self::Emerging),
            14 => Some(Self::Digging),
            15 => Some(Self::Sliding),
            16 => Some(Self::Shooting),
            17 => Some(Self::Inhaling),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            EntityPose::Standing => "standing",
            EntityPose::Gliding => "fall_flying",
            EntityPose::Sleeping => "sleeping",
            EntityPose::Swimming => "swimming",
            EntityPose::SpinAttack => "spin_attack",
            EntityPose::Crouching => "crouching",
            EntityPose::LongJumping => "long_jumping",
            EntityPose::Dying => "dying",
            EntityPose::Croaking => "croaking",
            EntityPose::UsingTongue => "using_tongue",
            EntityPose::Sitting => "sitting",
            EntityPose::Roaring => "roaring",
            EntityPose::Sniffing => "sniffing",
            EntityPose::Emerging => "emerging",
            EntityPose::Digging => "digging",
            EntityPose::Sliding => "sliding",
            EntityPose::Shooting => "shooting",
            EntityPose::Inhaling => "inhaling",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ArmPose {
    Empty,
    Item,
    Block,
    BowAndArrow,
    ThrowSpear,
    CrossbowHold,
    Spyglass,
    TootHorn,
    Brush,
}
