use azalea::{ecs::{entity::Entity, world::World}, entity::metadata::{Aggressive, DrownedConversion}};

use super::biped::BipedRenderState;

pub struct ZombieRenderState {
    pub parent: BipedRenderState,
    pub attacking: bool,
    pub converting_in_water: bool,
}

impl ZombieRenderState {
    pub fn new(world: &mut World, entity: Entity) -> Self {
        Self {
            parent: BipedRenderState::new(world, entity),
            attacking: world.get::<Aggressive>(entity).unwrap().0,
            converting_in_water: world.get::<DrownedConversion>(entity).unwrap().0,
        }
    }
}
