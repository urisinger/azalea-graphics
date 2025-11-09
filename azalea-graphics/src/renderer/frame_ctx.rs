use ash::vk::{self};
use vk_mem::MemoryUsage;

use crate::renderer::{
    vulkan::{
        buffer::Buffer, context::VkContext, frame_sync::FrameSync, object::VkObject,
        timestamp::TimestampQueryPool,
    },
    world_renderer::WorldRendererConfig,
};

pub struct FrameCtx<'a> {
    pub ctx: &'a VkContext,
    pub cmd: vk::CommandBuffer,
    pub image_index: u32,
    pub extent: vk::Extent2D,
    pub view_proj: glam::Mat4,
    pub camera_pos: glam::Vec3,
    pub frame_index: usize,
    pub config: WorldRendererConfig,
    pub timestamps: Option<&'a TimestampQueryPool>,
    pub frame_sync: &'a mut FrameSync,
}

impl FrameCtx<'_> {
    /// Upload data to a buffer using a staging buffer that is automatically
    /// deleted.
    pub fn upload_to<T>(&mut self, data: &[T], dst: &Buffer) {
        let mut staging = Buffer::new(
            self.ctx,
            dst.size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::Auto,
            true,
        );

        staging.upload_data(self.ctx, 0, data);

        unsafe {
            self.ctx.device().cmd_copy_buffer(
                self.cmd,
                staging.buffer,
                dst.buffer,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(dst.size)],
            );
        }
        self.delete(staging);
    }

    /// Upload data to an image using a staging buffer that is automatically
    /// deleted.
    pub fn upload_to_image<T>(
        &mut self,
        data: &[T],
        dst: vk::Image,
        layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        let size = (std::mem::size_of::<T>() * data.len()) as vk::DeviceSize;
        let mut staging = Buffer::new(
            self.ctx,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::Auto,
            true,
        );
        staging.upload_data(self.ctx, 0, data);

        unsafe {
            self.ctx.device().cmd_copy_buffer_to_image(
                self.cmd,
                staging.buffer,
                dst,
                layout,
                regions,
            );
        }

        self.delete(staging);
    }

    pub fn pipeline_barrier(
        &self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        buffer_barriers: &[vk::BufferMemoryBarrier],
        image_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.ctx.device().cmd_pipeline_barrier(
                self.cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                buffer_barriers,
                image_barriers,
            );
        }
    }

    pub fn delete<O: VkObject + 'static>(&mut self, obj: O) {
        self.frame_sync
            .add_to_deletion_queue(self.frame_index, Box::new(obj));
    }
}
