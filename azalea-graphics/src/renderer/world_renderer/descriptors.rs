use ash::{vk, Device};

use crate::renderer::vulkan::texture::Texture;

pub fn create_world_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let sampler_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(std::slice::from_ref(&sampler_binding));

    unsafe { device.create_descriptor_set_layout(&info, None).unwrap() }
}

pub fn create_world_descriptor_pool(device: &Device) -> vk::DescriptorPool {
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1);

    let info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(std::slice::from_ref(&pool_size))
        .max_sets(1);

    unsafe { device.create_descriptor_pool(&info, None).unwrap() }
}

pub fn allocate_world_descriptor_set(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> vk::DescriptorSet {
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(std::slice::from_ref(&layout));

    unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap()[0] }
}

pub fn update_world_texture_descriptor(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    tex: &Texture,
) {
    let image_info = vk::DescriptorImageInfo {
        sampler: tex.sampler,
        image_view: tex.view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };

    let write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(&image_info));

    unsafe {
        device.update_descriptor_sets(std::slice::from_ref(&write), &[]);
    }
}

pub struct Descriptors {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
}

impl Descriptors {
    pub fn new(device: &Device, texture: &Texture) -> Self {
        let layout = create_world_descriptor_set_layout(device);
        let pool = create_world_descriptor_pool(device);
        let set = allocate_world_descriptor_set(device, pool, layout);
        update_world_texture_descriptor(device, set, texture);
        Self { layout, pool, set }
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}


