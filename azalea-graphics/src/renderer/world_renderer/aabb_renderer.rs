use ash::{vk, Device};
use std::mem;

use crate::renderer::{
    vulkan::{buffer::Buffer, context::VkContext, frame_sync::MAX_FRAMES_IN_FLIGHT},
    world_renderer::types::VisibilityPushConstants,
};

const AABB_DEBUG_VERT: &[u8] = include_bytes!(env!("AABB_DEBUG_VERT"));
const AABB_DEBUG_FRAG: &[u8] = include_bytes!(env!("AABB_DEBUG_FRAG"));

pub struct AabbRenderer {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl AabbRenderer {
    pub fn new(ctx: &VkContext, render_pass: vk::RenderPass) -> Self {
        let device = ctx.device();

        // Descriptor set layout for visibility buffer
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&binding));
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None).unwrap() };

        // Pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(mem::size_of::<VisibilityPushConstants>() as u32);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None).unwrap() };

        // Create pipeline
        let pipeline = Self::create_pipeline(ctx, render_pass, pipeline_layout);

        // Create empty descriptor pool (will be populated when visibility buffers are created)
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(std::slice::from_ref(&pool_size))
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() };

        Self {
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets: Vec::new(),
        }
    }

    fn create_pipeline(ctx: &VkContext, render_pass: vk::RenderPass, pipeline_layout: vk::PipelineLayout) -> vk::Pipeline {
        let device = ctx.device();

        // Shader modules
        let vert_code = ash::util::read_spv(&mut std::io::Cursor::new(AABB_DEBUG_VERT)).unwrap();
        let frag_code = ash::util::read_spv(&mut std::io::Cursor::new(AABB_DEBUG_FRAG)).unwrap();

        let vert_module = unsafe {
            let info = vk::ShaderModuleCreateInfo::default().code(&vert_code);
            device.create_shader_module(&info, None).unwrap()
        };
        let frag_module = unsafe {
            let info = vk::ShaderModuleCreateInfo::default().code(&frag_code);
            device.create_shader_module(&info, None).unwrap()
        };

        let entry_point = std::ffi::CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&entry_point),
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
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

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
                .create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        pipeline
    }

    pub fn recreate_descriptor_sets(&mut self, device: &Device, visibility_buffers: &[Buffer; MAX_FRAMES_IN_FLIGHT]) {
        // Reset the descriptor pool to clear all sets
        unsafe {
            device.reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
        }
        
        // Allocate new descriptor sets (exactly MAX_FRAMES_IN_FLIGHT)
        let layouts = vec![self.descriptor_set_layout; MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        
        self.descriptor_sets = unsafe { 
            device.allocate_descriptor_sets(&alloc_info).unwrap() 
        };

        // Update each descriptor set with its corresponding buffer
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
        push_constants: &VisibilityPushConstants,
        instance_count: u32,
        buffer_index: usize,
    ) {

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    push_constants as *const VisibilityPushConstants as *const u8,
                    mem::size_of::<VisibilityPushConstants>(),
                ),
            );

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_sets[buffer_index]),
                &[],
            );

            // Draw 24 vertices (12 lines) per instance, instanced for each chunk
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
