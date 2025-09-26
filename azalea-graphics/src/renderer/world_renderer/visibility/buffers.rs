use std::{array::from_fn, sync::Arc};

use ash::vk;

use crate::renderer::vulkan::{
    buffer::Buffer, context::VkContext, frame_sync::MAX_FRAMES_IN_FLIGHT,
};

pub struct VisibilitySnapshot {
    pub radius: i32,
    pub height: i32,
    pub data: Vec<u32>,
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

    pub fn is_visible(&self, dx: i32, dy: i32, dz: i32) -> bool {
        self.index(dx, dy, dz)
            .map(|i| {
                if self.data[i] != 0 {
                   true 
                } else {
                    false
                }
            })
            .unwrap_or_else(|| {
                println!("{dx} {dy} {dz}");
                false
            })
    }
}

pub struct VisibilityBuffers {
    pub outputs: [Buffer; MAX_FRAMES_IN_FLIGHT],
    pub readbacks: [Buffer; MAX_FRAMES_IN_FLIGHT],
    pub radius: i32,
    pub height: i32,
    pub entry_count: usize,
    pub byte_size: vk::DeviceSize,

    cached: [Option<Arc<VisibilitySnapshot>>; MAX_FRAMES_IN_FLIGHT],
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
            cached: Default::default(),
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
        self.invalidate_all();
    }

    pub fn copy_to_readback(&self, ctx: &VkContext, cmd: vk::CommandBuffer, frame_idx: usize) {
        self.outputs[frame_idx].copy_to(ctx, &self.readbacks[frame_idx], cmd);
    }

    pub fn snapshot(&mut self, ctx: &VkContext, frame_idx: usize) -> Arc<VisibilitySnapshot> {
        if let Some(cached) = &self.cached[frame_idx] {
            return Arc::clone(cached);
        }

        let allocator = ctx.allocator();
        let mut data = vec![0u32; self.entry_count];
        unsafe {
            let ptr = allocator
                .map_memory(&mut self.readbacks[frame_idx].allocation)
                .unwrap();
            std::ptr::copy_nonoverlapping(ptr as *const u32, data.as_mut_ptr(), self.entry_count);
            allocator.unmap_memory(&mut self.readbacks[frame_idx].allocation);
        }
        let snap = Arc::new(VisibilitySnapshot {
            radius: self.radius,
            height: self.height,
            data,
        });
        self.cached[frame_idx] = Some(Arc::clone(&snap));
        snap
    }

    pub fn invalidate(&mut self, frame_idx: usize) {
        self.cached[frame_idx] = None;
    }

    pub fn invalidate_all(&mut self) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            self.cached[i] = None;
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        for b in &mut self.outputs {
            b.destroy(ctx);
        }
        for b in &mut self.readbacks {
            b.destroy(ctx);
        }
        self.invalidate_all();
    }
}
