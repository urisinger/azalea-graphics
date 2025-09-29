use ash::{Device, vk};

use crate::renderer::{
    vulkan::{buffer::Buffer, context::VkContext, frame_sync::MAX_FRAMES_IN_FLIGHT},
    world_renderer::{
        hiz::HiZPyramid, types::VisibilityPushConstants, visibility::buffers::VisibilityBuffers,
    },
};

pub struct VisibilityCompute {
    pub layout_frame: vk::DescriptorSetLayout,
    pub layout_image: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub pool_frame: vk::DescriptorPool,
    pub pool_image: vk::DescriptorPool,

    pub sets_frame: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    pub sets_image: Vec<vk::DescriptorSet>,

    pub radius: i32,
    pub height: i32,
}

impl VisibilityCompute {
    pub fn new(
        ctx: &VkContext,
        module: vk::ShaderModule,
        pyramids: &[HiZPyramid],
        radius: i32,
        height: i32,
    ) -> Self {
        let d = ctx.device();
        let images = pyramids.len();

        let frame_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let layout_frame = unsafe {
            d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&frame_bindings),
                None,
            )
            .unwrap()
        };

        // set=1 (per-image): b0 Hi-Z sampler2D
        let image_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let layout_image = unsafe {
            d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&image_bindings),
                None,
            )
            .unwrap()
        };

        let entry = std::ffi::CString::new("visibility::compute_visibility").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);

        println!("{}", std::mem::size_of::<VisibilityPushConstants>() as u32);
        let pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<VisibilityPushConstants>() as u32);

        let set_layouts = [layout_frame, layout_image];
        let pipeline_layout = unsafe {
            d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&pc_range)),
                None,
            )
            .unwrap()
        };
        let pipeline = unsafe {
            d.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            )
            .unwrap()[0]
        };
        unsafe { d.destroy_shader_module(module, None) };

        // pools
        let pool_frame = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
                    }])
                    .max_sets(MAX_FRAMES_IN_FLIGHT as u32),
                None,
            )
            .unwrap()
        };
        let pool_image = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: images as u32,
                    }])
                    .max_sets(images as u32),
                None,
            )
            .unwrap()
        };

        // allocate sets
        let sets_frame = {
            let layouts = [layout_frame; MAX_FRAMES_IN_FLIGHT];
            let flat = unsafe {
                d.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool_frame)
                        .set_layouts(&layouts),
                )
                .unwrap()
            };
            let mut arr = [vk::DescriptorSet::null(); MAX_FRAMES_IN_FLIGHT];
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                arr[i] = flat[i];
            }
            arr
        };
        let sets_image = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool_image)
                    .set_layouts(&vec![layout_image; images]),
            )
            .unwrap()
        };

        // write per-image sets (set=1 → b0 = hiz sampler)
        for i in 0..images {
            let hiz = vk::DescriptorImageInfo {
                sampler: pyramids[i].sampler,
                image_view: pyramids[i].full_view, // full pyramid view
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            let write = vk::WriteDescriptorSet::default()
                .dst_set(sets_image[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&hiz));
            unsafe { d.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }

        Self {
            layout_frame,
            layout_image,
            pipeline_layout,
            pipeline,
            pool_frame,
            pool_image,
            sets_frame,
            sets_image,
            radius,
            height,
        }
    }

    /// Bind sets and push the camera data as push constants; no descriptor
    /// updates during dispatch.

    pub fn dispatch(
        &self,
        ctx: &VkContext,
        cmd: vk::CommandBuffer,
        image_index: usize,
        frame_index: usize,
        vis_buffers: &VisibilityBuffers,
        pc: &VisibilityPushConstants,
    ) {
        let d = ctx.device();
        let side = (vis_buffers.radius * 2 + 1) as u32;
        let h = vis_buffers.height as u32;

        unsafe {
            // Run compute shader
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            let sets = [self.sets_frame[frame_index], self.sets_image[image_index]];
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &sets,
                &[],
            );
            d.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(pc),
            );
            d.cmd_dispatch(cmd, side, h, side);

            // Barrier to make sure compute writes are visible to transfer
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(vis_buffers.outputs[frame_index].buffer)
                    .offset(0)
                    .size(vis_buffers.byte_size)],
                &[],
            );

            d.cmd_copy_buffer(
                cmd,
                vis_buffers.outputs[frame_index].buffer,
                vis_buffers.readbacks[frame_index].buffer,
                &[vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(vis_buffers.byte_size)],
            );
        }
    }

    pub fn rewrite_frame_set(&self, device: &Device, frame_index: usize, output_buffer: &Buffer) {
        let out = vk::DescriptorBufferInfo {
            buffer: output_buffer.buffer,
            offset: 0,
            range: output_buffer.size,
        };
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.sets_frame[frame_index])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&out));
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }

    pub fn recreate_image_sets(&mut self, ctx: &VkContext, pyramids: &[HiZPyramid]) {
        let d = ctx.device();
        unsafe { d.destroy_descriptor_pool(self.pool_image, None) };
        let images = pyramids.len();

        self.pool_image = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: images as u32,
                    }])
                    .max_sets(images as u32),
                None,
            )
            .unwrap()
        };
        self.sets_image = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool_image)
                    .set_layouts(&vec![self.layout_image; images]),
            )
            .unwrap()
        };
        for i in 0..images {
            let hiz = vk::DescriptorImageInfo {
                sampler: pyramids[i].sampler,
                image_view: pyramids[i].full_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.sets_image[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&hiz));
            unsafe { d.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        unsafe {
            let d = ctx.device();
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_pool(self.pool_frame, None);
            d.destroy_descriptor_pool(self.pool_image, None);
            d.destroy_descriptor_set_layout(self.layout_frame, None);
            d.destroy_descriptor_set_layout(self.layout_image, None);
        }
    }
}
