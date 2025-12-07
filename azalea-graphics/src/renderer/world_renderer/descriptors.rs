use std::array::from_fn;

use ash::{Device, vk};

use crate::renderer::vulkan::{buffer::Buffer, frame_sync::MAX_FRAMES_IN_FLIGHT, texture::Texture};

pub fn create_world_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
    let sampler_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
    ];

    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&sampler_bindings);

    unsafe { device.create_descriptor_set_layout(&info, None).unwrap() }
}

pub fn create_world_descriptor_pool(device: &Device) -> vk::DescriptorPool {
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
    ];

    let info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

    unsafe { device.create_descriptor_pool(&info, None).unwrap() }
}

pub fn allocate_world_descriptor_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT] {
    let layouts = [layout; MAX_FRAMES_IN_FLIGHT];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);

    let sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

    sets.try_into().unwrap()
}

pub fn update_world_texture_descriptor(
    device: &Device,
    descriptor_sets: &[vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    uniform_buffers: &[Buffer; MAX_FRAMES_IN_FLIGHT],
    tex: &Texture,
) {
    let image_info = vk::DescriptorImageInfo {
        sampler: tex.sampler,
        image_view: tex.view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };

    let mut writes = Vec::new();

    let buffer_infos: [_; MAX_FRAMES_IN_FLIGHT] = from_fn(|i| {
        vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffers[i].buffer)
            .range(vk::WHOLE_SIZE)
    });

    for i in 0..MAX_FRAMES_IN_FLIGHT {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info)),
        );

        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[i])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[i])),
        )
    }

    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

pub struct Descriptors {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
}

impl Descriptors {
    pub fn new(
        device: &Device,
        uniform_buffers: &[Buffer; MAX_FRAMES_IN_FLIGHT],
        texture: &Texture,
    ) -> Self {
        let layout = create_world_descriptor_set_layout(device);
        let pool = create_world_descriptor_pool(device);
        let sets = allocate_world_descriptor_sets(device, pool, layout);
        update_world_texture_descriptor(device, &sets, uniform_buffers, texture);
        Self { layout, pool, sets }
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
