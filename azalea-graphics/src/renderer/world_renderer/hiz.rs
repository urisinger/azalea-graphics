use ash::{Device, vk};
use vk_mem::Alloc;

use crate::renderer::{
    frame_ctx::FrameCtx,
    vulkan::{context::VkContext, image::AllocatedImage},
    world_renderer::render_targets::RenderTargets,
};

pub struct HiZPyramid {
    pub image: vk::Image,
    pub allocation: vk_mem::Allocation,
    pub sampler: vk::Sampler,
    pub mip_levels: u32,
    pub mip_views: Vec<vk::ImageView>,
    pub full_view: vk::ImageView,
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

pub struct HiZCompute {
    pub copy_layout: vk::DescriptorSetLayout,
    pub reduce_layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub copy_sets: Vec<vk::DescriptorSet>,
    pub reduce_sets: Vec<Vec<vk::DescriptorSet>>,
    pub copy_pipeline_layout: vk::PipelineLayout,
    pub reduce_pipeline_layout: vk::PipelineLayout,
    pub copy_pipeline: vk::Pipeline,
    pub reduce_pipeline: vk::Pipeline,
    pub frames: usize,
    pub mip_levels: u32,
    pub depth_sampler: vk::Sampler,
}

impl HiZCompute {
    pub fn new(
        ctx: &VkContext,
        module: vk::ShaderModule,
        pyramids: &[HiZPyramid],
        depth_images: &[AllocatedImage],
    ) -> Self {
        assert!(!pyramids.is_empty());
        assert_eq!(pyramids.len(), depth_images.len());

        let frames = pyramids.len();
        let mip_levels = pyramids[0].mip_levels;

        let copy_bindings = [
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
        let copy_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&copy_bindings);
        let copy_layout = unsafe {
            ctx.device()
                .create_descriptor_set_layout(&copy_layout_info, None)
                .unwrap()
        };

        let reduce_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let reduce_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&reduce_bindings);
        let reduce_layout = unsafe {
            ctx.device()
                .create_descriptor_set_layout(&reduce_layout_info, None)
                .unwrap()
        };

        let copy_pipeline_layout = unsafe {
            let pli = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&copy_layout));
            ctx.device().create_pipeline_layout(&pli, None).unwrap()
        };

        let reduce_pipeline_layout = unsafe {
            let pli = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&reduce_layout));
            ctx.device().create_pipeline_layout(&pli, None).unwrap()
        };

        let copy_pipeline = create_compute_pipeline(ctx, module, "hiz::copy", copy_pipeline_layout);
        let reduce_pipeline =
            create_compute_pipeline(ctx, module, "hiz::reduce", reduce_pipeline_layout);

        let (pool, copy_sets, reduce_sets) =
            Self::alloc_sets(ctx, copy_layout, reduce_layout, frames, mip_levels);

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
            copy_layout,
            reduce_layout,
            pool,
            copy_sets,
            reduce_sets,
            copy_pipeline_layout,
            reduce_pipeline_layout,
            copy_pipeline,
            reduce_pipeline,
            frames,
            mip_levels,
            depth_sampler,
        };

        this.recreate_descriptors(ctx.device(), pyramids, depth_images);
        this
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
            let (pool, copy_sets, reduce_sets) = Self::alloc_sets(
                ctx,
                self.copy_layout,
                self.reduce_layout,
                new_frames,
                new_mips,
            );
            self.pool = pool;
            self.copy_sets = copy_sets;
            self.reduce_sets = reduce_sets;
            self.frames = new_frames;
            self.mip_levels = new_mips;
        }

        self.recreate_descriptors(ctx.device(), pyramids, depth_images);
    }

    fn alloc_sets(
        ctx: &VkContext,
        copy_layout: vk::DescriptorSetLayout,
        reduce_layout: vk::DescriptorSetLayout,
        frames: usize,
        mip_levels: u32,
    ) -> (
        vk::DescriptorPool,
        Vec<vk::DescriptorSet>,
        Vec<Vec<vk::DescriptorSet>>,
    ) {
        let copy_total = frames;
        let reduce_total = frames * (mip_levels as usize - 1);

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(copy_total as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count((copy_total + reduce_total * 2) as u32),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&sizes)
            .max_sets((copy_total + reduce_total) as u32);
        let pool = unsafe {
            ctx.device()
                .create_descriptor_pool(&pool_info, None)
                .unwrap()
        };

        let copy_layouts = vec![copy_layout; copy_total];
        let copy_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&copy_layouts);
        let copy_sets = unsafe { ctx.device().allocate_descriptor_sets(&copy_alloc).unwrap() };

        let reduce_layouts = vec![reduce_layout; reduce_total];
        let reduce_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&reduce_layouts);
        let reduce_flat = unsafe {
            ctx.device()
                .allocate_descriptor_sets(&reduce_alloc)
                .unwrap()
        };

        let mut reduce_sets = Vec::with_capacity(frames);
        let per_frame = mip_levels as usize - 1;
        for f in 0..frames {
            let s = f * per_frame;
            let e = s + per_frame;
            reduce_sets.push(reduce_flat[s..e].to_vec());
        }

        (pool, copy_sets, reduce_sets)
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
            assert_eq!(pyr.mip_levels, self.mip_levels);

            let src = vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: depth_images[f].default_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            let dst = vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: pyr.mip_views[0],
                image_layout: vk::ImageLayout::GENERAL,
            };

            let copy_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.copy_sets[f])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(&src)),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.copy_sets[f])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&dst)),
            ];
            unsafe { device.update_descriptor_sets(&copy_writes, &[]) };

            for level in 1..self.mip_levels {
                let src = vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: pyr.mip_views[(level - 1) as usize],
                    image_layout: vk::ImageLayout::GENERAL,
                };
                let dst = vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: pyr.mip_views[level as usize],
                    image_layout: vk::ImageLayout::GENERAL,
                };

                let set = self.reduce_sets[f][(level - 1) as usize];
                let reduce_writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(&src)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(&dst)),
                ];
                unsafe { device.update_descriptor_sets(&reduce_writes, &[]) };
            }
        }
    }

    pub fn dispatch_all_levels(&self, frame_ctx: &mut FrameCtx, render_targets: &RenderTargets) {
        let FrameCtx {
            ctx,
            cmd,
            image_index,
            extent,
            ..
        } = frame_ctx;
        let device = ctx.device();
        let pyramid = &render_targets.depth_pyramids[*image_index as usize];

        let pyramid_full = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: pyramid.mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        };

        unsafe {
            device.cmd_pipeline_barrier(
                *cmd,
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

            device.cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, self.copy_pipeline);
            device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.copy_pipeline_layout,
                0,
                &[self.copy_sets[*image_index as usize]],
                &[],
            );
            let gx = (extent.width + 7) / 8;
            let gy = (extent.height + 7) / 8;
            device.cmd_dispatch(*cmd, gx.max(1), gy.max(1), 1);

            let mip0_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            device.cmd_pipeline_barrier(
                *cmd,
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

            device.cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::COMPUTE, self.reduce_pipeline);
        }

        let mut w = (extent.width / 2).max(1);
        let mut h = (extent.height / 2).max(1);

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
                    *cmd,
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
                        .subresource_range(prev_range)],
                );
                device.cmd_bind_descriptor_sets(
                    *cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.reduce_pipeline_layout,
                    0,
                    &[self.reduce_sets[*image_index as usize][(level - 1) as usize]],
                    &[],
                );
                let gx = (w + 7) / 8;
                let gy = (h + 7) / 8;
                device.cmd_dispatch(*cmd, gx.max(1), gy.max(1), 1);
            }

            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }

        unsafe {
            device.cmd_pipeline_barrier(
                *cmd,
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
            d.destroy_pipeline_layout(self.copy_pipeline_layout, None);
            d.destroy_pipeline_layout(self.reduce_pipeline_layout, None);
            d.destroy_descriptor_pool(self.pool, None);
            d.destroy_descriptor_set_layout(self.copy_layout, None);
            d.destroy_descriptor_set_layout(self.reduce_layout, None);
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
