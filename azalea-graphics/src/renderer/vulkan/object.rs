use crate::renderer::vulkan::context::VkContext;

pub trait VkObject {
    fn destroy(&self, ctx: &VkContext);
}
