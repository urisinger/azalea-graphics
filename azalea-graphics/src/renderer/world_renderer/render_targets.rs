use ash::{Device, vk};

use crate::renderer::{
    vulkan::{context::VkContext, image::AllocatedImage, swapchain::Swapchain},
    world_renderer::hiz::HiZPyramid,
};

pub struct RenderTargets {
    pub render_pass: vk::RenderPass,
    pub depth_images: Vec<AllocatedImage>,
    pub depth_pyramids: Vec<HiZPyramid>,
    pub mip_levels: u32,
    pub framebuffers: Vec<vk::Framebuffer>,
}

impl RenderTargets {
    pub fn new(ctx: &VkContext, swapchain: &Swapchain) -> Self {
        let render_pass = create_world_render_pass(ctx, swapchain);
        let depth_images = create_world_depth_resources(ctx, swapchain);

        let depth_pyramids: Vec<_> = (0..swapchain.image_views.len())
            .map(|_| HiZPyramid::new(ctx, swapchain.extent.width, swapchain.extent.height))
            .collect();

        let mip_levels = depth_pyramids.first().map(|p| p.mip_levels).unwrap_or(1);

        let framebuffers = create_world_framebuffers(ctx, swapchain, render_pass, &depth_images);

        Self {
            render_pass,
            depth_images,
            depth_pyramids,
            mip_levels,
            framebuffers,
        }
    }

    pub fn recreate(&mut self, ctx: &VkContext, swapchain: &Swapchain) {
        self.destory_frame_resources(ctx);

        self.depth_images = create_world_depth_resources(ctx, swapchain);
        self.depth_pyramids = (0..swapchain.image_views.len())
            .map(|_| HiZPyramid::new(ctx, swapchain.extent.width, swapchain.extent.height))
            .collect();
        self.mip_levels = self
            .depth_pyramids
            .first()
            .map(|p| p.mip_levels)
            .unwrap_or(1);

        self.framebuffers =
            create_world_framebuffers(ctx, swapchain, self.render_pass, &self.depth_images);
    }

    #[inline]
    pub fn framebuffer(&self, index: usize) -> vk::Framebuffer {
        self.framebuffers[index]
    }

    pub fn destory_frame_resources(&mut self, ctx: &VkContext) {
        let device = ctx.device();

        for fb in self.framebuffers.drain(..) {
            unsafe { device.destroy_framebuffer(fb, None) };
        }

        for pyramid in &mut self.depth_pyramids {
            pyramid.destroy(ctx);
        }
        self.depth_pyramids.clear();

        for img in &mut self.depth_images {
            img.destroy(ctx);
        }
        self.depth_images.clear();
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        self.destory_frame_resources(ctx);
        unsafe { ctx.device().destroy_render_pass(self.render_pass, None) };
    }

    pub fn begin(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        image_index: u32,
        extent: vk::Extent2D,
    ) {
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
        ];

        let rp_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer(image_index as usize))
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values);

        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_info, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );
        }
    }

    pub fn end(&self, device: &Device, cmd: vk::CommandBuffer) {
        unsafe { device.cmd_end_render_pass(cmd) };
    }
}

pub fn create_world_render_pass(ctx: &VkContext, swapchain: &Swapchain) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription::default()
        .format(swapchain.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_attachment = vk::AttachmentDescription::default()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let color_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let depth_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_stage_mask(
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .dependency_flags(vk::DependencyFlags::BY_REGION),
    ];

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref))
        .depth_stencil_attachment(&depth_ref);

    let attachments = [color_attachment, depth_attachment];

    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&dependencies);

    unsafe { ctx.device().create_render_pass(&info, None).unwrap() }
}

pub fn create_world_depth_resources(ctx: &VkContext, swapchain: &Swapchain) -> Vec<AllocatedImage> {
    let format = vk::Format::D32_SFLOAT;
    (0..swapchain.image_views.len())
        .map(|_| {
            AllocatedImage::depth_2d_device(
                ctx,
                format,
                swapchain.extent.width,
                swapchain.extent.height,
                vk::SampleCountFlags::TYPE_1,
                vk::ImageUsageFlags::SAMPLED,
            )
        })
        .collect()
}

pub fn create_world_framebuffers(
    ctx: &VkContext,
    swapchain: &Swapchain,
    render_pass: vk::RenderPass,
    depth_images: &[AllocatedImage],
) -> Vec<vk::Framebuffer> {
    let device = ctx.device();
    let mut fbs = Vec::with_capacity(swapchain.image_views.len());

    for (i, &color_view) in swapchain.image_views.iter().enumerate() {
        let depth_view = depth_images[i].default_view;
        let attachments = [color_view, depth_view];

        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(swapchain.extent.width)
            .height(swapchain.extent.height)
            .layers(1);

        let fb = unsafe { device.create_framebuffer(&info, None).unwrap() };
        fbs.push(fb);
    }
    fbs
}
