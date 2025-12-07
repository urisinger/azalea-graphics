use std::ffi::CString;

use ash::{Device, vk};

use crate::renderer::{
    vulkan::{buffer::Buffer, context::VkContext, frame_sync::MAX_FRAMES_IN_FLIGHT},
    world_renderer::types::VisibilityUniform,
};

pub struct AabbRenderer {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
}

impl AabbRenderer {
    pub fn new(
        ctx: &VkContext,
        uniform_buffers: &[Buffer; MAX_FRAMES_IN_FLIGHT],
        module: vk::ShaderModule,
        render_pass: vk::RenderPass,
    ) -> Self {
        let device = ctx.device();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let pipeline = Self::create_pipeline(ctx, module, render_pass, pipeline_layout);

        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() };
        let layouts = vec![descriptor_set_layout; MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets: [_; MAX_FRAMES_IN_FLIGHT] =
            unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() }
                .try_into()
                .expect("size should match frames in flight");

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_sets[i])
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(uniform_buffers[i].buffer)
                            .range(size_of::<VisibilityUniform>() as u64)])],
                    &[],
                );
            }
        }

        Self {
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
        }
    }

    fn create_pipeline(
        ctx: &VkContext,
        module: vk::ShaderModule,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
    ) -> vk::Pipeline {
        let device = ctx.device();

        let vert_entry = CString::new("debug::aabb_vert").unwrap();
        let frag_entry = CString::new("debug::aabb_frag").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(module)
                .name(&vert_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(module)
                .name(&frag_entry),
        ];

        // No vertex input (geometry generated in shader)
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::LINE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::LINE)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        pipeline
    }

    pub fn recreate_descriptor_sets(
        &mut self,
        device: &Device,
        visibility_buffers: &[Buffer; MAX_FRAMES_IN_FLIGHT],
    ) {
        for (i, buffer) in visibility_buffers.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));

            unsafe {
                device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
            }
        }
    }

    pub fn draw(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        instance_count: u32,
        buffer_index: usize,
    ) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_sets[buffer_index]),
                &[],
            );

            device.cmd_draw(cmd, 24, instance_count, 0, 0);
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
