use ash::{vk, Device};
use crate::renderer::{vulkan::context::VkContext, world_renderer::{types::BlockVertex, types::PushConstants}};

fn create_shader_module(device: &Device, code: &[u8]) -> vk::ShaderModule {
    let code_aligned = ash::util::read_spv(&mut std::io::Cursor::new(code)).unwrap();
    let info = vk::ShaderModuleCreateInfo::default().code(&code_aligned);
    unsafe { device.create_shader_module(&info, None).unwrap() }
}

pub fn create_world_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> vk::PipelineLayout {
    let layouts = [descriptor_set_layout];
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<PushConstants>() as u32);

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&layouts)
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));

    unsafe { device.create_pipeline_layout(&pipeline_layout_info, None).unwrap() }
}

pub struct PipelineConfig {
    pub polygon_mode: vk::PolygonMode,
    pub enable_blend: bool,
    pub depth_write: bool,
}

pub fn create_world_pipeline(
    ctx: &VkContext,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
    config: PipelineConfig,
) -> vk::Pipeline {
    let device = ctx.device();

    let vert_module = create_shader_module(device, vert_spv);
    let frag_module = create_shader_module(device, frag_spv);

    let entry_point = std::ffi::CString::new("main").unwrap();

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(&entry_point),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(&entry_point),
    ];

    let binding_desc = [BlockVertex::binding_description()];
    let attribute_desc = BlockVertex::attribute_descriptions();

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_desc)
        .vertex_attribute_descriptions(&attribute_desc);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(config.polygon_mode)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let mut color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(config.enable_blend);

    if config.enable_blend {
        color_blend_attachment = color_blend_attachment
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
    }

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(config.depth_write)
        .depth_compare_op(vk::CompareOp::LESS);

    let attachments = [color_blend_attachment];
    let color_blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blending)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipelines = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .expect("Failed to create pipeline")
    };
    let pipeline = pipelines[0];

    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }

    pipeline
}

pub struct Pipelines {
    pub layout: vk::PipelineLayout,
    pub block: vk::Pipeline,
    pub block_wire: Option<vk::Pipeline>,
    pub water: vk::Pipeline,
    pub water_wire: Option<vk::Pipeline>,
}

pub struct PipelineOptions {
    pub wireframe_enabled: bool,
}

impl Pipelines {
    pub fn new(
        ctx: &VkContext,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
        block_vert_spv: &[u8],
        block_frag_spv: &[u8],
        water_vert_spv: &[u8],
        water_frag_spv: &[u8],
        opts: PipelineOptions,
    ) -> Self {
        let layout = create_world_pipeline_layout(ctx.device(), descriptor_set_layout);

        let block = create_world_pipeline(
            ctx,
            render_pass,
            layout,
            block_vert_spv,
            block_frag_spv,
            super::pipelines::PipelineConfig { polygon_mode: vk::PolygonMode::FILL, enable_blend: false, depth_write: true },
        );
        let block_wire = if opts.wireframe_enabled {
            Some(create_world_pipeline(
                ctx,
                render_pass,
                layout,
                block_vert_spv,
                block_frag_spv,
                super::pipelines::PipelineConfig { polygon_mode: vk::PolygonMode::LINE, enable_blend: false, depth_write: true },
            ))
        } else { None };

        let water = create_world_pipeline(
            ctx,
            render_pass,
            layout,
            water_vert_spv,
            water_frag_spv,
            super::pipelines::PipelineConfig { polygon_mode: vk::PolygonMode::FILL, enable_blend: true, depth_write: false },
        );
        let water_wire = if opts.wireframe_enabled {
            Some(create_world_pipeline(
                ctx,
                render_pass,
                layout,
                water_vert_spv,
                water_frag_spv,
                super::pipelines::PipelineConfig { polygon_mode: vk::PolygonMode::LINE, enable_blend: true, depth_write: false },
            ))
        } else { None };

        Self { layout, block, block_wire, water, water_wire }
    }

    pub fn block_pipeline(&self, wireframe_mode: bool) -> vk::Pipeline {
        if wireframe_mode { self.block_wire.unwrap_or(self.block) } else { self.block }
    }
    pub fn water_pipeline(&self, wireframe_mode: bool) -> vk::Pipeline {
        if wireframe_mode { self.water_wire.unwrap_or(self.water) } else { self.water }
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            if let Some(p) = self.block_wire.take() { device.destroy_pipeline(p, None); }
            if let Some(p) = self.water_wire.take() { device.destroy_pipeline(p, None); }
            device.destroy_pipeline(self.block, None);
            device.destroy_pipeline(self.water, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
