use std::num::NonZero;

use azalea::{
    app::{App, AppExit, Plugin, Update},
    block_update::{QueuedServerBlockUpdates, handle_block_update_event},
    chunks::{ReceiveChunkEvent, handle_receive_chunk_event},
    core::position::{ChunkPos, ChunkSectionPos},
    ecs::{
        entity::Entity,
        message::{MessageReader, MessageWriter},
        query::Changed,
        schedule::IntoScheduleConfigs,
        system::{Query, Res, SystemState},
        world::World,
    },
    entity::EntityKindComponent,
    local_player::InstanceHolder,
    prelude::*,
};
use crossbeam::channel::TryRecvError;

use crate::{app::{RendererEvent, RendererHandle}, renderer::RenderState};

#[derive(Resource, Clone)]
pub struct RendererResource {
    pub handle: RendererHandle,
}

pub struct RendererPlugin {
    pub handle: RendererHandle,
}

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RendererResource {
            handle: self.handle.clone(),
        });
        app.add_systems(
            Update,
            forward_chunk_updates
                .after(handle_receive_chunk_event)
                .after(handle_block_update_event),
        );
        app.add_systems(Update, add_world.before(forward_chunk_updates));
        app.add_systems(
            Update,
            handle_block_updates.before(handle_block_update_event),
        );
        app.add_systems(Update, get_entities);
        app.add_systems(Update, poll_renderer_events);
    }
}

pub fn handle_block_updates(
    query: Query<(&QueuedServerBlockUpdates, &InstanceHolder)>,
    renderer: Res<RendererResource>,
) {
    for (queued, instance_holder) in query.iter() {
        let world = instance_holder.instance.read();
        for (pos, block_state) in &queued.list {
            renderer.handle.send_section(ChunkSectionPos::from(pos));
        }
    }
}

fn forward_chunk_updates(
    mut events: MessageReader<ReceiveChunkEvent>,
    renderer: Res<RendererResource>,
) {
    for event in events.read() {
        let pos = ChunkPos::new(event.packet.x, event.packet.z);

        renderer.handle.send_chunk(pos)
    }
}

fn add_world(
    renderer: Res<RendererResource>,
    added: Query<&InstanceHolder, Changed<InstanceHolder>>,
) {
    for holder in added {
        println!("added");
        renderer.handle.add_world(holder.instance.clone());
    }
}

fn get_entities(
    world: &mut World,
    params: &mut SystemState<(Res<RendererResource>, Query<(Entity, &EntityKindComponent)>)>,
) {
    let mut entites = Vec::new();

    let (renderer, entity_kinds) = params.get(world);
    let entities_mutex = renderer.handle.entities.clone();
    let entity_kinds = entity_kinds
        .iter()
        .map(|(entity, entity_kind)| (entity, entity_kind.clone()))
        .collect::<Vec<_>>();
    for (entity, entity_kind) in entity_kinds {
        if let Some(e) = RenderState::from_entity(world, entity_kind.0, entity) {
            entites.push(e);
        }
    }

    *entities_mutex.lock() = entites;
}

fn poll_renderer_events(renderer: Res<RendererResource>, mut writer: MessageWriter<AppExit>) {
    match renderer.handle.rx.try_recv() {
        Ok(RendererEvent::Closed) => {
            writer.write(AppExit::Success);
        }
        Err(TryRecvError::Empty) => {}
        Err(TryRecvError::Disconnected) => {
            writer.write(AppExit::Error(NonZero::new(1).unwrap()));
        }
    }
}
