use std::{collections::HashMap, sync::Arc};

use ash::{vk, Device};
use azalea_assets::Assets;

use crate::renderer::{
    frame_ctx::FrameCtx,
    vulkan::{context::VkContext, frame_sync::MAX_FRAMES_IN_FLIGHT, texture::Texture},
};

const MAX_TEXTURES: u32 = 1024;

pub struct TextureManager {
    assets: Arc<Assets>,
    textures: Vec<Texture>,
    name_to_index: HashMap<String, u32>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
    dirty_descriptor_sets: [bool; MAX_FRAMES_IN_FLIGHT],
}

impl TextureManager {
    pub fn new(ctx: &VkContext, assets: Arc<Assets>) -> Self {
        let descriptor_set_layout = Self::create_descriptor_set_layout(ctx.device());
        let descriptor_pool = Self::create_descriptor_pool(ctx.device());
        let descriptor_sets =
            Self::allocate_descriptor_sets(ctx.device(), descriptor_pool, descriptor_set_layout);

        Self {
            assets,
            textures: Vec::new(),
            name_to_index: HashMap::new(),
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            dirty_descriptor_sets: [true; MAX_FRAMES_IN_FLIGHT],
        }
    }

    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_TEXTURES)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe { device.create_descriptor_set_layout(&info, None).unwrap() }
    }

    fn create_descriptor_pool(device: &Device) -> vk::DescriptorPool {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_TEXTURES * MAX_FRAMES_IN_FLIGHT as u32)];

        let info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

        unsafe { device.create_descriptor_pool(&info, None).unwrap() }
    }

    fn allocate_descriptor_sets(
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

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn get_descriptor_set(&mut self, device: &Device, frame_index: usize) -> vk::DescriptorSet {
        // Update descriptor set if it's dirty
        if self.dirty_descriptor_sets[frame_index] {
            self.update_descriptor_set(device, frame_index);
            self.dirty_descriptor_sets[frame_index] = false;
        }

        self.descriptor_sets[frame_index]
    }

    pub fn get_texture(&mut self, ctx: &mut FrameCtx, id: &str) -> u32 {
        if let Some(&texture_id) = self.name_to_index.get(id) {
            texture_id
        } else {
            let path = self.assets.get_path(id);
            let image = if let Ok(image) = image::open(path) {
                image
            } else {
                return 0;
            };
            let image = if let Some(image) = image.as_rgba8() {
                image
            } else {
                return 0;
            };

            let (width, height) = image.dimensions();
            let mut texture = Texture::new(
                ctx.ctx,
                width,
                height,
                vk::Filter::NEAREST,
                vk::Filter::NEAREST,
            );
            texture.upload_data(ctx, image.as_raw(), width, height);

            let texture_id = self.textures.len() as u32;
            
            self.textures.push(texture);
            self.name_to_index.insert(id.to_string(), texture_id);
            
            // Mark all descriptor sets as dirty since we added a new texture
            for dirty in &mut self.dirty_descriptor_sets {
                *dirty = true;
            }
            
            texture_id
        }
    }

    fn update_descriptor_set(&self, device: &Device, frame_index: usize) {
        if self.textures.is_empty() {
            return;
        }

        let mut image_infos = Vec::with_capacity(self.textures.len());
        for texture in &self.textures {
            image_infos.push(vk::DescriptorImageInfo {
                sampler: texture.sampler,
                image_view: texture.view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            });
        }

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_sets[frame_index])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);

        unsafe {
            device.update_descriptor_sets(&[write], &[]);
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        // Destroy all textures
        for texture in &mut self.textures {
            texture.destroy(ctx);
        }
        
        // Destroy descriptor resources
        let device = ctx.device();
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
