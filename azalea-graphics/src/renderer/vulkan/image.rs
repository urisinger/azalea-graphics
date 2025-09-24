use ash::{Device, vk};
use vk_mem::{Alloc, Allocation, AllocationCreateInfo, MemoryUsage};

use crate::renderer::vulkan::context::VkContext;

pub struct AllocatedImage {
    pub image: vk::Image,
    pub allocation: Allocation,

    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub usage: vk::ImageUsageFlags,

    pub default_view: vk::ImageView,
}

impl AllocatedImage {
    pub fn new_2d_with_view(
        ctx: &VkContext,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
        array_layers: u32,
        samples: vk::SampleCountFlags,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        memory_usage: MemoryUsage,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Self {
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .samples(samples)
            .tiling(tiling)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let alloc_info = AllocationCreateInfo {
            usage: memory_usage,
            ..Default::default()
        };

        let (image, allocation) = unsafe {
            ctx.allocator()
                .create_image(&image_info, &alloc_info)
                .expect("Failed to create VMA image")
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(if array_layers > 1 {
                vk::ImageViewType::TYPE_2D_ARRAY
            } else {
                vk::ImageViewType::TYPE_2D
            })
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: array_layers,
            });

        let default_view = unsafe { ctx.device().create_image_view(&view_info, None).unwrap() };

        Self {
            image,
            allocation,
            format,
            extent,
            mip_levels,
            array_layers,
            samples,
            usage,
            default_view,
        }
    }

    pub fn color_2d_device(
        ctx: &VkContext,
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        Self::new_2d_with_view(
            ctx,
            format,
            width,
            height,
            mip_levels,
            1,
            vk::SampleCountFlags::TYPE_1,
            vk::ImageTiling::OPTIMAL,
            usage,
            MemoryUsage::AutoPreferDevice,
            vk::ImageAspectFlags::COLOR,
        )
    }

    pub fn depth_2d_device(
        ctx: &VkContext,
        format: vk::Format,
        width: u32,
        height: u32,
        samples: vk::SampleCountFlags,
        extra_usage: vk::ImageUsageFlags,
    ) -> Self {
        let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | extra_usage;
        let aspect = if format == vk::Format::D32_SFLOAT || format == vk::Format::D16_UNORM {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        };
        Self::new_2d_with_view(
            ctx,
            format,
            width,
            height,
            1,
            1,
            samples,
            vk::ImageTiling::OPTIMAL,
            usage,
            MemoryUsage::AutoPreferDevice,
            aspect,
        )
    }

    pub fn create_mip_view(
        &self,
        device: &Device,
        aspect_mask: vk::ImageAspectFlags,
        mip: u32,
    ) -> vk::ImageView {
        assert!(mip < self.mip_levels);
        let info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: mip,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        unsafe { device.create_image_view(&info, None).unwrap() }
    }

    pub fn create_view_range(
        &self,
        device: &Device,
        view_type: vk::ImageViewType,
        aspect_mask: vk::ImageAspectFlags,
        base_mip_level: u32,
        level_count: u32,
        base_array_layer: u32,
        layer_count: u32,
    ) -> vk::ImageView {
        let info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(view_type)
            .format(self.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level,
                level_count,
                base_array_layer,
                layer_count,
            });
        unsafe { device.create_image_view(&info, None).unwrap() }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        unsafe {
            let device = ctx.device();
            device.destroy_image_view(self.default_view, None);
            ctx.allocator()
                .destroy_image(self.image, &mut self.allocation);
        }
    }
}
