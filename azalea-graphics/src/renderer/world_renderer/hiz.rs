use ash::{Device, vk};
use vk_mem::Alloc;

use crate::renderer::vulkan::{context::VkContext, image::AllocatedImage};

pub struct HiZPyramid {
    pub image: vk::Image,
    pub allocation: vk_mem::Allocation,
    pub sampler: vk::Sampler,
    pub mip_levels: u32,
    pub mip_views: Vec<vk::ImageView>,
    pub full_view: vk::ImageView,
}

pub struct HiZCompute {
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub sets: Vec<Vec<vk::DescriptorSet>>,
    pub pipeline_layout: vk::PipelineLayout,
    pub copy_pipeline: vk::Pipeline,
    pub reduce_pipeline: vk::Pipeline,
    pub frames: usize,
    pub mip_levels: u32,
    pub depth_sampler: vk::Sampler,
}

impl HiZPyramid {
    pub fn new(ctx: &VkContext, width: u32, height: u32) -> Self {
        let max_dim = width.max(height).max(1);
        let mip_levels = (u32::BITS - max_dim.leading_zeros()) as u32;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        let (image, allocation) = unsafe {
            ctx.allocator()
                .create_image(&image_info, &alloc_info)
                .unwrap()
        };

        ctx.label_object(image, "HiZ Image");
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(mip_levels as f32);
        let sampler = unsafe { ctx.device().create_sampler(&sampler_info, None).unwrap() };

        let full_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });
        let full_view = unsafe {
            ctx.device()
                .create_image_view(&full_view_info, None)
                .unwrap()
        };

        let mut mip_views = Vec::with_capacity(mip_levels as usize);
        for level in 0..mip_levels {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: level,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = unsafe { ctx.device().create_image_view(&view_info, None).unwrap() };
            mip_views.push(view);
        }

        Self {
            image,
            allocation,
            sampler,
            mip_levels,
            mip_views,
            full_view,
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        unsafe {
            let d = ctx.device();

            d.destroy_image_view(self.full_view, None);
            for v in self.mip_views.drain(..) {
                d.destroy_image_view(v, None);
            }
            d.destroy_sampler(self.sampler, None);
            ctx.allocator()
                .destroy_image(self.image, &mut self.allocation);
        }
    }
}

impl HiZCompute {
    pub fn new(
        ctx: &VkContext,
        module: vk::ShaderModule,
        pyramids: &[HiZPyramid],
        depth_images: &[AllocatedImage],
    ) -> Self {
        assert!(!pyramids.is_empty(), "need at least one HiZPyramid");
        assert_eq!(
            pyramids.len(),
            depth_images.len(),
            "pyramids vs depth images mismatch"
        );

        let frames = pyramids.len();
        let mip_levels = pyramids[0].mip_levels;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = unsafe {
            ctx.device()
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        };

        let pipeline_layout = unsafe {
            let pli =
                vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&layout));
            ctx.device().create_pipeline_layout(&pli, None).unwrap()
        };

        let copy_pipeline = create_compute_pipeline(ctx, module, "visibility::hiz_copy", pipeline_layout);
        let reduce_pipeline = create_compute_pipeline(ctx, module, "visibility::hiz_reduce", pipeline_layout);

        let (pool, sets) = Self::alloc_sets(ctx, layout, frames, mip_levels);

        let depth_sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(0.0);
        let depth_sampler = unsafe {
            ctx.device()
                .create_sampler(&depth_sampler_info, None)
                .unwrap()
        };

        let this = Self {
            layout,
            pool,
            sets,
            pipeline_layout,
            copy_pipeline,
            reduce_pipeline,
            frames,
            mip_levels,
            depth_sampler,
        };

        this.recreate_descriptors(ctx.device(), pyramids, depth_images);

        this
    }

    fn alloc_sets(
        ctx: &VkContext,
        layout: vk::DescriptorSetLayout,
        frames: usize,
        mip_levels: u32,
    ) -> (vk::DescriptorPool, Vec<Vec<vk::DescriptorSet>>) {
        let total = frames * mip_levels as usize;

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(total as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(total as u32),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&sizes)
            .max_sets(total as u32);
        let pool = unsafe {
            ctx.device()
                .create_descriptor_pool(&pool_info, None)
                .unwrap()
        };

        let layouts = vec![layout; total];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let flat = unsafe { ctx.device().allocate_descriptor_sets(&alloc).unwrap() };

        let mut sets = Vec::with_capacity(frames);
        for f in 0..frames {
            let s = f * mip_levels as usize;
            let e = s + mip_levels as usize;
            sets.push(flat[s..e].to_vec());
        }
        (pool, sets)
    }

    pub fn recreate_descriptors(
        &self,
        device: &Device,
        pyramids: &[HiZPyramid],
        depth_images: &[AllocatedImage],
    ) {
        assert_eq!(pyramids.len(), self.frames);
        assert_eq!(depth_images.len(), self.frames);

        for f in 0..self.frames {
            let pyr = &pyramids[f];
            assert_eq!(
                pyr.mip_levels, self.mip_levels,
                "mip count mismatch at frame {f}"
            );

            for level in 0..self.mip_levels {
                let (src, dst) = if level == 0 {
                    (
                        vk::DescriptorImageInfo {
                            sampler: self.depth_sampler,
                            image_view: depth_images[f].default_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: pyr.mip_views[0],
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                    )
                } else {
                    (
                        vk::DescriptorImageInfo {
                            sampler: pyr.sampler,
                            image_view: pyr.mip_views[(level - 1) as usize],
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: pyr.mip_views[level as usize],
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                    )
                };

                let set = self.sets[f][level as usize];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(&src)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(&dst)),
                ];
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
        }
    }

    pub fn recreate(
        &mut self,
        ctx: &VkContext,
        pyramids: &[HiZPyramid],
        depth_images: &[AllocatedImage],
    ) {
        assert!(!pyramids.is_empty());
        assert_eq!(pyramids.len(), depth_images.len());

        let new_frames = pyramids.len();
        let new_mips = pyramids[0].mip_levels;

        if new_frames != self.frames || new_mips != self.mip_levels {
            unsafe { ctx.device().destroy_descriptor_pool(self.pool, None) };
            let (pool, sets) = Self::alloc_sets(ctx, self.layout, new_frames, new_mips);
            self.pool = pool;
            self.sets = sets;
            self.frames = new_frames;
            self.mip_levels = new_mips;
        }

        self.recreate_descriptors(ctx.device(), pyramids, depth_images);
    }

    pub fn dispatch_all_levels(
        &self,
        ctx: &VkContext,
        cmd: vk::CommandBuffer,
        image_idx: u32,
        pyramid: &HiZPyramid,
        depth_image: &AllocatedImage,
        base_width: u32,
        base_height: u32,
    ) {
        let device = ctx.device();

        let depth_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let pyramid_full = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: pyramid.mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(pyramid.image)
                    .subresource_range(pyramid_full)],
            );
        }

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.copy_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.sets[image_idx as usize][0]],
                &[],
            );
            let gx = (base_width + 7) / 8;
            let gy = (base_height + 7) / 8;
            device.cmd_dispatch(cmd, gx.max(1), gy.max(1), 1);
        }

        let mip0_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(pyramid.image)
                    .subresource_range(mip0_range)],
            );
        }

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.reduce_pipeline);
        }

        let mut w = (base_width / 2).max(1);
        let mut h = (base_height / 2).max(1);

        for level in 1..pyramid.mip_levels {
            let prev = level - 1;
            let prev_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: prev,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER, // prev write happened in compute
                    vk::PipelineStageFlags::COMPUTE_SHADER, // this read happens in compute
                    vk::DependencyFlags::BY_REGION,
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE) // prev wrote
                        .dst_access_mask(vk::AccessFlags::SHADER_READ) // now we read it
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .image(pyramid.image)
                        .subresource_range(prev_range)],
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[self.sets[image_idx as usize][level as usize]],
                    &[],
                );
                let gx = (w + 7) / 8;
                let gy = (h + 7) / 8;
                device.cmd_dispatch(cmd, gx.max(1), gy.max(1), 1);
            }

            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(pyramid.image)
                    .subresource_range(pyramid_full)],
            );
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        unsafe {
            let d = ctx.device();
            d.destroy_sampler(self.depth_sampler, None);
            d.destroy_pipeline(self.copy_pipeline, None);
            d.destroy_pipeline(self.reduce_pipeline, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

fn create_compute_pipeline(
    ctx: &VkContext,
    module: vk::ShaderModule,
    entry: &str,
    pipeline_layout: vk::PipelineLayout,
) -> vk::Pipeline {
    unsafe {
        let entry = std::ffi::CString::new(entry).unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);
        let ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline = ctx
            .device()
            .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&ci), None)
            .unwrap()[0];
        pipeline
    }
}
