use azalea::{
    ecs::{
        entity::{Entity},
        world::World,
    },
    registry::EntityKind,
};
use zombie::ZombieRenderState;

mod armed_entity;
mod biped;
mod entity;
mod living_entity;
mod zombie;

pub enum RenderState {
    Zombie(ZombieRenderState),
}

impl RenderState {
    pub fn from_entity(
        world: &mut World,
        entity_kind: EntityKind,
        entity: Entity,
    ) -> Option<Self> {
        match entity_kind {
            EntityKind::Zombie => Some(Self::Zombie(ZombieRenderState::new(world, entity))),
            _ => None,
        }
    }
}
