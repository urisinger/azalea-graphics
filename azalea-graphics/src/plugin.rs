use std::num::NonZero;

use azalea::{
    app::{App, AppExit, Plugin, Update},
    block_update::{handle_block_update_event, QueuedServerBlockUpdates},
    chunks::{handle_receive_chunk_event, ReceiveChunkEvent},
    core::position::{ChunkPos, ChunkSectionPos},
    ecs::{
        event::{EventReader, EventWriter},
        query::Changed,
        schedule::IntoScheduleConfigs,
        system::{Query, Res},
    },
    local_player::InstanceHolder,
    prelude::*,
};
use crossbeam::channel::TryRecvError;

use crate::app::{RendererEvent, RendererHandle};

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
        app.add_systems(Update, handle_block_updates.before(handle_block_update_event));
        app.add_systems(Update, poll_renderer_events);
    }
}

pub fn handle_block_updates(
    query: Query<(
        &QueuedServerBlockUpdates,
        &InstanceHolder,
    )>,

    renderer: Res<RendererResource>,
) {
    for (queued, instance_holder) in query.iter() {
        let world = instance_holder.instance.read();
        for (pos, block_state) in &queued.list {
            let pos = pos.with_y(pos.y - world.chunks.min_y);
            renderer.handle.send_section(ChunkSectionPos::from(pos));
        }
    }
}

fn forward_chunk_updates(
    mut events: EventReader<ReceiveChunkEvent>,
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

fn poll_renderer_events(renderer: Res<RendererResource>, mut writer: EventWriter<AppExit>) {
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
