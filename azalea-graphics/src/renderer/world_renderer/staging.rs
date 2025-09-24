use crate::renderer::vulkan::{buffer::Buffer, frame_sync::MAX_FRAMES_IN_FLIGHT};

use super::super::vulkan::context::VkContext;

pub struct StagingArena {
    pub per_frame: [Vec<Buffer>; MAX_FRAMES_IN_FLIGHT],
}

impl Default for StagingArena {
    fn default() -> Self {
        Self {
            per_frame: Default::default(),
        }
    }
}

impl StagingArena {
    pub fn clear_frame(&mut self, ctx: &VkContext, frame_index: usize) {
        for mut buffer in self.per_frame[frame_index].drain(..) {
            buffer.destroy(ctx);
        }
    }

    pub fn push(&mut self, frame_index: usize, buffer: Buffer) {
        self.per_frame[frame_index].push(buffer);
    }

    pub fn destroy_all(&mut self, ctx: &VkContext) {
        for buffers in &mut self.per_frame {
            for mut buffer in buffers.drain(..) {
                buffer.destroy(ctx);
            }
        }
    }
}


