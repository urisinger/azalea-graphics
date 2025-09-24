use ash::{Device, vk};
use vk_mem::{Alloc, Allocation, AllocationCreateInfo, MemoryUsage};

use crate::renderer::vulkan::{context::VkContext, swapchain::Swapchain};

pub struct RenderTargets {
    pub render_pass: vk::RenderPass,
    pub depth_images: Vec<(vk::Image, Allocation, vk::ImageView)>,
    pub depth_pyramids: Vec<(vk::Image, Allocation, vk::ImageView)>,
    pub pyramid_views: Vec<Vec<vk::ImageView>>,
    pub mip_levels: u32,
    pub framebuffers: Vec<vk::Framebuffer>,
}

impl RenderTargets {
    pub fn new(ctx: &VkContext, swapchain: &Swapchain) -> Self {
        let render_pass = create_world_render_pass(ctx, swapchain);
        let depth_images = create_world_depth_resources(ctx, swapchain);
        let (depth_pyramids, pyramid_views, mip_levels) = create_depth_pyramids(ctx, swapchain);
        let framebuffers = create_world_framebuffers(ctx, swapchain, render_pass, &depth_images);

        Self {
            render_pass,
            depth_images,
            depth_pyramids,
            pyramid_views,
            mip_levels,
            framebuffers,
        }
    }

    pub fn recreate(&mut self, ctx: &VkContext, swapchain: &Swapchain) {
        self.destory_frame_resources(ctx);
        self.depth_images = create_world_depth_resources(ctx, swapchain);
        let (depth_pyramids, pyramid_views, mip_levels) = create_depth_pyramids(ctx, swapchain);
        self.depth_pyramids = depth_pyramids;
        self.pyramid_views = pyramid_views;
        self.mip_levels = mip_levels;
        self.framebuffers =
            create_world_framebuffers(ctx, swapchain, self.render_pass, &self.depth_images);
    }

    pub fn framebuffer(&self, index: usize) -> vk::Framebuffer {
        self.framebuffers[index]
    }

    pub fn destory_frame_resources(&mut self, ctx: &VkContext) {
        let device = ctx.device();
        for fb in self.framebuffers.drain(..) {
            unsafe { device.destroy_framebuffer(fb, None) };
        }
        for views in &self.pyramid_views {
            for view in views {
                unsafe {
                    device.destroy_image_view(*view, None);
                }
            }
        }
        for (image, mut alloc, view) in self.depth_pyramids.drain(..) {
            unsafe {
                device.destroy_image_view(view, None);
                ctx.allocator().destroy_image(image, &mut alloc);
            }
        }
        for (image, mut alloc, view) in self.depth_images.drain(..) {
            unsafe {
                device.destroy_image_view(view, None);
                ctx.allocator().destroy_image(image, &mut alloc);
            }
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        self.destory_frame_resources(ctx);
        unsafe {
            ctx.device().destroy_render_pass(self.render_pass, None);
        }
    }

    pub fn begin(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        image_index: u32,
        extent: vk::Extent2D,
    ) {
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let clear_values = [
            clear_color,
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
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
        }

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        unsafe {
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
        }
    }

    pub fn end(&self, device: &Device, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_end_render_pass(cmd);
        }
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
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

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
            .src_stage_mask(
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
    ];

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref))
        .depth_stencil_attachment(&depth_ref);

    let attachments = [color_attachment, depth_attachment];

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&dependencies);

    unsafe {
        ctx.device()
            .create_render_pass(&render_pass_info, None)
            .unwrap()
    }
}

pub fn create_world_depth_resources(
    ctx: &VkContext,
    swapchain: &Swapchain,
) -> Vec<(vk::Image, Allocation, vk::ImageView)> {
    let format = vk::Format::D32_SFLOAT;

    swapchain
        .image_views
        .iter()
        .map(|_| {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: swapchain.extent.width,
                    height: swapchain.extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let alloc_info = AllocationCreateInfo {
                usage: MemoryUsage::AutoPreferDevice,
                ..Default::default()
            };

            let (image, allocation) = unsafe {
                ctx.allocator()
                    .create_image(&image_info, &alloc_info)
                    .unwrap()
            };

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let depth_view = unsafe { ctx.device().create_image_view(&view_info, None).unwrap() };

            (image, allocation, depth_view)
        })
        .collect()
}

pub fn create_depth_pyramids(
    ctx: &VkContext,
    swapchain: &Swapchain,
) -> (
    Vec<(vk::Image, Allocation, vk::ImageView)>,
    Vec<Vec<vk::ImageView>>,
    u32,
) {
    let format = vk::Format::R32_SFLOAT;
    let max_dim = swapchain.extent.width.max(swapchain.extent.height);
    let mip_levels = (max_dim as f32).log2().floor() as u32 + 1;

    let mut pyramids = Vec::with_capacity(swapchain.image_views.len());
    let mut mip_views = Vec::with_capacity(swapchain.image_views.len());

    for _ in &swapchain.image_views {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: swapchain.extent.width,
                height: swapchain.extent.height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) = unsafe {
            ctx.allocator()
                .create_image(&image_info, &alloc_info)
                .expect("Failed to create Hi-Z image")
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        let full_view = unsafe { ctx.device().create_image_view(&view_info, None).unwrap() };
        pyramids.push((image, allocation, full_view));

        let mut this_mip_views = Vec::with_capacity(mip_levels as usize);
        for mip in 0..mip_levels {
            let mip_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let mip_view = unsafe {
                ctx.device()
                    .create_image_view(&mip_view_info, None)
                    .unwrap()
            };
            this_mip_views.push(mip_view);
        }
        mip_views.push(this_mip_views);
    }

    (pyramids, mip_views, mip_levels)
}

pub fn create_world_framebuffers(
    ctx: &VkContext,
    swapchain: &Swapchain,
    render_pass: vk::RenderPass,
    depth_resources: &[(vk::Image, Allocation, vk::ImageView)],
) -> Vec<vk::Framebuffer> {
    let device = ctx.device();
    let mut framebuffers = Vec::with_capacity(swapchain.image_views.len());

    for (i, &view) in swapchain.image_views.iter().enumerate() {
        let depth_view = depth_resources[i].2;
        let attachments = [view, depth_view];

        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(swapchain.extent.width)
            .height(swapchain.extent.height)
            .layers(1);

        let framebuffer = unsafe { device.create_framebuffer(&info, None).unwrap() };
        framebuffers.push(framebuffer);
    }

    framebuffers
}
