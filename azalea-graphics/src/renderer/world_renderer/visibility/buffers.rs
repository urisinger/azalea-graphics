use std::array::from_fn;

use ash::vk;
use azalea::core::position::ChunkSectionPos;

use crate::renderer::vulkan::{
    buffer::Buffer,
    context::VkContext,
    frame_sync::{FrameSync, MAX_FRAMES_IN_FLIGHT},
};

pub struct VisibilitySnapshot {
    pub radius: i32,
    pub height: i32,
    pub data: Vec<f32>,

    pub cx: i32,
    pub cz: i32,
    pub min_y: i32,
}

impl VisibilitySnapshot {
    pub fn index(&self, dx: i32, dy: i32, dz: i32) -> Option<usize> {
        let side = self.radius * 2 + 1;
        if dx < -self.radius || dx > self.radius {
            return None;
        }
        if dz < -self.radius || dz > self.radius {
            return None;
        }
        if dy < 0 || dy >= self.height {
            return None;
        }

        let x = (dx + self.radius) as usize;
        let z = (dz + self.radius) as usize;
        let y = dy as usize;

        Some((y * side as usize * side as usize) + (z * side as usize) + x)
    }

    pub fn get_depth(&self, dx: i32, dy: i32, dz: i32) -> Option<f32> {
        self.index(dx, dy, dz).map(|i| self.data[i])
    }

    pub fn is_visible(&self, dx: i32, dy: i32, dz: i32) -> bool {
        self.index(dx, dy, dz)
            .map(|i| self.data[i] != 0.0)
            .unwrap_or(false)
    }

    pub fn section_is_visible(&self, spos: ChunkSectionPos) -> bool {
        let dx = spos.x - self.cx;
        let dy = spos.y - (self.min_y / 16);
        let dz = spos.z - self.cz;
        self.is_visible(dx, dy, dz)
    }
    pub fn section_depth(&self, spos: ChunkSectionPos) -> Option<f32> {
        let dx = spos.x - self.cx;
        let dy = spos.y - (self.min_y / 16);
        let dz = spos.z - self.cz;
        self.get_depth(dx, dy, dz)
    }
}

pub struct VisibilityBuffers {
    pub outputs: [Buffer; MAX_FRAMES_IN_FLIGHT],
    pub readbacks: [Buffer; MAX_FRAMES_IN_FLIGHT],
    pub radius: i32,
    pub height: i32,
    pub entry_count: usize,
    pub byte_size: vk::DeviceSize,
}

impl VisibilityBuffers {
    fn calc(radius: i32, height: i32) -> (usize, vk::DeviceSize) {
        let side = (radius * 2 + 1) as usize;
        let count = side * side * height as usize;
        let bytes = (count * std::mem::size_of::<u32>()) as vk::DeviceSize;
        (count, bytes)
    }

    pub fn new(ctx: &VkContext, radius: i32, height: i32) -> Self {
        let (entry_count, byte_size) = Self::calc(radius, height);
        let outputs = from_fn(|_| {
            Buffer::new(
                ctx,
                byte_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                vk_mem::MemoryUsage::AutoPreferDevice,
                false,
            )
        });
        let readbacks = from_fn(|_| {
            Buffer::new(
                ctx,
                byte_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferHost,
                true,
            )
        });
        Self {
            outputs,
            readbacks,
            radius,
            height,
            entry_count,
            byte_size,
        }
    }

    pub fn recreate(&mut self, ctx: &VkContext, radius: i32, height: i32) {
        if self.radius == radius && self.height == height {
            return;
        }
        for b in &mut self.outputs {
            b.destroy(ctx);
        }
        for b in &mut self.readbacks {
            b.destroy(ctx);
        }
        let (entry_count, byte_size) = Self::calc(radius, height);
        self.outputs = std::array::from_fn(|_| {
            Buffer::new(
                ctx,
                byte_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                vk_mem::MemoryUsage::AutoPreferDevice,
                false,
            )
        });
        self.readbacks = std::array::from_fn(|_| {
            Buffer::new(
                ctx,
                byte_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferHost,
                true,
            )
        });
        self.radius = radius;
        self.height = height;
        self.entry_count = entry_count;
        self.byte_size = byte_size;
    }

    pub fn copy_to_readback(&self, ctx: &VkContext, cmd: vk::CommandBuffer, frame_idx: usize) {
        self.outputs[frame_idx].copy_to(ctx, &self.readbacks[frame_idx], cmd);
    }

    pub fn snapshot(
        &mut self,
        ctx: &VkContext,
        frame_idx: usize,
        cx: i32,
        cz: i32,
        min_y: i32,
    ) -> VisibilitySnapshot {
        let allocator = ctx.allocator();
        let mut data = vec![0.0; self.entry_count];
        unsafe {
            let ptr = allocator
                .map_memory(&mut self.readbacks[frame_idx].allocation)
                .unwrap();
            std::ptr::copy_nonoverlapping(ptr as *const f32, data.as_mut_ptr(), self.entry_count);
            allocator.unmap_memory(&mut self.readbacks[frame_idx].allocation);
        }
        VisibilitySnapshot {
            radius: self.radius,
            height: self.height,
            cx,
            cz,
            data,
            min_y,
        }
    }

    pub fn resize(
        &mut self,
        ctx: &VkContext,
        sync: &mut FrameSync,
        new_radius: i32,
        new_height: i32,
    ) {
        if self.radius == new_radius && self.height == new_height {
            return;
        }

        for (frame, b) in &mut self.outputs.iter().enumerate() {
            sync.add_to_deletion_queue(frame, Box::new(b.clone()));
        }
        for (frame, b) in &mut self.readbacks.iter().enumerate() {
            sync.add_to_deletion_queue(frame, Box::new(b.clone()));
        }

        let (entry_count, byte_size) = Self::calc(new_radius, new_height);

        self.outputs = std::array::from_fn(|_| {
            Buffer::new(
                ctx,
                byte_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                vk_mem::MemoryUsage::AutoPreferDevice,
                false,
            )
        });

        self.readbacks = std::array::from_fn(|_| {
            Buffer::new(
                ctx,
                byte_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk_mem::MemoryUsage::AutoPreferHost,
                true,
            )
        });

        self.radius = new_radius;
        self.height = new_height;
        self.entry_count = entry_count;
        self.byte_size = byte_size;
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        for b in &mut self.outputs {
            b.destroy(ctx);
        }
        for b in &mut self.readbacks {
            b.destroy(ctx);
        }
    }
}
