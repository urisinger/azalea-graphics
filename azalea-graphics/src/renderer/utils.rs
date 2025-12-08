use ash::vk;

use crate::renderer::{render_targets::RenderTargets, vulkan::context::VkContext};

pub fn create_framebuffers(
    ctx: &VkContext,
    render_targets: &RenderTargets,
    render_pass: vk::RenderPass,
) -> Vec<vk::Framebuffer> {
    let device = ctx.device();
    let mut fbs = Vec::with_capacity(render_targets.swapchain.image_views.len());

    for (i, &color_view) in render_targets.swapchain.image_views.iter().enumerate() {
        let depth_view = render_targets.depth_images[i].default_view;
        let attachments = [color_view, depth_view];

        let info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(render_targets.swapchain.extent.width)
            .height(render_targets.swapchain.extent.height)
            .layers(1);

        let fb = unsafe { device.create_framebuffer(&info, None).unwrap() };
        fbs.push(fb);
    }
    fbs
}


