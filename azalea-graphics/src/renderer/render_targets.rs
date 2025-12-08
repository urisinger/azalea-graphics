use ash::vk;

use crate::renderer::{
    hiz::HiZPyramid,
    vulkan::{context::VkContext, image::AllocatedImage, swapchain::Swapchain},
};

pub struct RenderTargets {
    pub depth_images: Vec<AllocatedImage>,
    pub depth_pyramids: Vec<HiZPyramid>,
    pub mip_levels: u32,
    pub swapchain: Swapchain,
}

impl RenderTargets {
    pub fn new(ctx: &VkContext, width: u32, height: u32) -> Self {
        let swapchain = Swapchain::new(ctx, width, height);
        let depth_images = create_depth_resources(ctx, &swapchain);

        let depth_pyramids: Vec<_> = (0..swapchain.image_views.len())
            .map(|_| HiZPyramid::new(ctx, swapchain.extent.width, swapchain.extent.height))
            .collect();

        let mip_levels = depth_pyramids.first().map(|p| p.mip_levels).unwrap_or(1);

        Self {
            depth_images,
            depth_pyramids,
            mip_levels,
            swapchain,
        }
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    pub fn recreate(&mut self, ctx: &VkContext, width: u32, height: u32) {
        self.swapchain.recreate(ctx, width, height);
        self.destory_frame_resources(ctx);

        self.depth_images = create_depth_resources(ctx, &self.swapchain);
        self.depth_pyramids = (0..self.swapchain.image_views.len())
            .map(|_| HiZPyramid::new(ctx, width, height))
            .collect();
        self.mip_levels = self
            .depth_pyramids
            .first()
            .map(|p| p.mip_levels)
            .unwrap_or(1);
    }

    pub fn destory_frame_resources(&mut self, ctx: &VkContext) {
        let device = ctx.device();

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
        self.swapchain.destroy(&ctx.device());
        self.destory_frame_resources(ctx);
    }
}

pub fn create_depth_resources(ctx: &VkContext, swapchain: &Swapchain) -> Vec<AllocatedImage> {
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
