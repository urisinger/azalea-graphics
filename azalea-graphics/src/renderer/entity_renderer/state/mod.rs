use std::ops::Deref;

use azalea::{
    ecs::{entity::Entity, world::World},
    registry::EntityKind,
};
use zombie::ZombieRenderState;

use crate::renderer::entity_renderer::state::entity::EntityRenderState;

pub mod armed_entity;
pub mod biped;
pub mod entity;
pub mod living_entity;
pub mod zombie;

pub enum RenderState {
    Zombie(ZombieRenderState),
}

impl Deref for RenderState {
    type Target = EntityRenderState;
    fn deref(&self) -> &Self::Target {
        match self {
            RenderState::Zombie(s) => &s,
        }
    }
}

impl RenderState {
    pub fn from_entity(world: &mut World, entity_kind: EntityKind, entity: Entity) -> Option<Self> {
        match entity_kind {
            EntityKind::Zombie => Some(Self::Zombie(ZombieRenderState::new(world, entity))),
            _ => None,
        }
    }
}
