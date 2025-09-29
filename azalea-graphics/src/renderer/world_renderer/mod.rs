use std::{cmp::Ordering, collections::HashMap, sync::Arc};

use ash::vk;
use azalea::core::position::ChunkSectionPos;
use glam::Vec3;
use image::GenericImageView;
use vk_mem::MemoryUsage;

use crate::{
    app::WorldUpdate,
    renderer::{
        assets::{Assets, processed::atlas::TextureEntry},
        vulkan::{
            buffer::Buffer, context::VkContext, frame_sync::MAX_FRAMES_IN_FLIGHT,
            swapchain::Swapchain, texture::Texture,
        },
        world_renderer::{
            aabb_renderer::AabbRenderer,
            animation::AnimationManager,
            mesher::Mesher,
            visibility::{buffers::VisibilityBuffers, compute::VisibilityCompute},
        },
    },
};

mod aabb_renderer;
mod animation;
mod descriptors;
mod hiz;
mod mesher;
mod meshes;
mod pipelines;
mod render_targets;
mod staging;
mod types;
mod visibility;

use descriptors::Descriptors;
use meshes::MeshStore;
use pipelines::{PipelineOptions, Pipelines};
use render_targets::RenderTargets;
use staging::StagingArena;
use types::{BlockVertex, PushConstants, VisibilityPushConstants};

pub struct WorldRenderer {
    pub mesher: Option<Mesher>,

    animation_manager: AnimationManager,
    mesh_store: MeshStore,

    hiz_compute: hiz::HiZCompute,
    visibility_compute: VisibilityCompute,
    visibility_buffers: Option<VisibilityBuffers>,
    aabb_renderer: AabbRenderer,

    pipelines: Pipelines,
    descriptors: Descriptors,
    render_targets: RenderTargets,
    blocks_texture: Texture,
    staging: StagingArena,
    assets: Arc<Assets>,
}

pub struct WorldRendererFeatures {
    pub fill_mode_non_solid: bool,
}

impl Default for WorldRendererFeatures {
    fn default() -> Self {
        Self {
            fill_mode_non_solid: false,
        }
    }
}

#[derive(Clone, Copy)]
pub struct WorldRendererConfig {
    pub wireframe_mode: bool,
    pub render_aabbs: bool,
}

impl Default for WorldRendererConfig {
    fn default() -> Self {
        Self {
            wireframe_mode: false,
            render_aabbs: false,
        }
    }
}

impl WorldRenderer {
    pub fn new(
        assets: Arc<Assets>,
        ctx: &VkContext,
        module: vk::ShaderModule,
        swapchain: &Swapchain,
        options: WorldRendererFeatures,
    ) -> Self {
        let atlas_image = animation::create_initial_atlas(&assets.block_atlas, &assets.textures);
        let blocks_texture = Texture::new(ctx, atlas_image);

        let descriptors = Descriptors::new(ctx.device(), &blocks_texture);

        let render_targets = RenderTargets::new(ctx, swapchain);

        let pipelines = Pipelines::new(
            ctx,
            render_targets.render_pass,
            descriptors.layout,
            module,
            PipelineOptions {
                wireframe_enabled: options.fill_mode_non_solid,
            },
        );

        let hiz_compute = hiz::HiZCompute::new(
            ctx,
            module,
            &render_targets.depth_pyramids,
            &render_targets.depth_images,
        );

        let visibility_compute =
            VisibilityCompute::new(ctx, module, &render_targets.depth_pyramids, 32, 1);
        let aabb_renderer = AabbRenderer::new(ctx, module, render_targets.render_pass);

        Self {
            mesher: None,
            animation_manager: AnimationManager::from_textures(&assets.textures),
            hiz_compute,

            visibility_compute,
            visibility_buffers: None,
            aabb_renderer,

            staging: Default::default(),
            mesh_store: Default::default(),
            pipelines,
            descriptors,
            render_targets,
            blocks_texture,
            assets: assets.clone(),
        }
    }

    pub fn tick(&mut self) {
        self.animation_manager.tick(&self.assets.textures);
    }

    pub fn update_visibility(&mut self, ctx: &VkContext, frame_index: usize, camera_pos: Vec3) {
        if let (Some(mesher), Some(vis_bufs)) = (&self.mesher, &mut self.visibility_buffers) {
            let cx = (camera_pos.x / 16.0).floor() as i32;
            let cy = (camera_pos.y / 16.0).floor() as i32;
            let cz = (camera_pos.z / 16.0).floor() as i32;
            let min_y = self.mesher.as_ref().unwrap().world.read().chunks.min_y;
            let snapshot = vis_bufs.snapshot(ctx, frame_index, cx, cy, cz, min_y);

            mesher.update_visibility(snapshot);
        }
    }

    pub fn update(&mut self, ctx: &VkContext, update: WorldUpdate) {
        match update {
            WorldUpdate::ChunkAdded(chunk_pos) => {
                if let Some(mesher) = &self.mesher {
                    mesher.submit_chunk(chunk_pos);
                }
            }
            WorldUpdate::SectionChange(spos) => {
                if let Some(mesher) = &self.mesher {
                    if let Some(vis) = &mut self.visibility_buffers {
                        mesher.submit_section(spos);
                    }
                }
            }
            WorldUpdate::WorldAdded(world) => {
                if let Some(mut vb) = self.visibility_buffers.take() {
                    vb.destroy(ctx);
                }

                let world_read = world.read();
                let max_height = world_read.chunks.height as i32 - world_read.chunks.min_y;
                drop(world_read);

                let vb = VisibilityBuffers::new(ctx, 32, (max_height) / 16);

                for f in 0..MAX_FRAMES_IN_FLIGHT {
                    self.visibility_compute
                        .rewrite_frame_set(ctx.device(), f, &vb.outputs[f]);
                }

                self.aabb_renderer
                    .recreate_descriptor_sets(ctx.device(), &vb.outputs);

                self.visibility_buffers = Some(vb);

                self.mesher = Some(Mesher::new(self.assets.clone(), world));
            }
        }
    }

    pub fn render(
        &mut self,
        ctx: &VkContext,
        cmd: vk::CommandBuffer,
        image_index: u32,
        extent: vk::Extent2D,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        frame_index: usize,
        state: WorldRendererConfig,
    ) {
        ctx.cmd_begin_debug_label(cmd, &format!("World Render Frame {}", frame_index));

        self.staging.clear_frame(ctx, frame_index);

        ctx.cmd_begin_debug_label(cmd, "Update meshes");
        self.mesh_store.process_mesher_results(
            ctx,
            cmd,
            frame_index,
            &self.mesher,
            &mut self.staging,
        );

        ctx.cmd_end_debug_label(cmd);

        ctx.cmd_begin_debug_label(cmd, "Update dirty textures");
        self.upload_dirty_textures(ctx, cmd, frame_index);
        ctx.cmd_end_debug_label(cmd);

        ctx.cmd_begin_debug_label(cmd, "Main Render Pass");
        self.render_targets
            .begin(ctx.device(), cmd, image_index, extent);
        self.draw(
            ctx,
            cmd,
            view_proj,
            camera_pos,
            frame_index,
            state.wireframe_mode,
        );

        let visibility_push_constants = if let Some(vb) = &mut self.visibility_buffers {
            const CHUNK: f32 = 16.0;

            let cam_chunk_x = (camera_pos.x / CHUNK).floor() as i32;
            let cam_chunk_z = (camera_pos.z / CHUNK).floor() as i32;
            let grid_min_x = (cam_chunk_x) as f32 * CHUNK;
            let grid_min_z = (cam_chunk_z) as f32 * CHUNK;
            let grid_origin_ws = [
                grid_min_x,
                (self
                    .mesher
                    .as_ref()
                    .map(|m| m.world.read().chunks.min_y)
                    .unwrap_or(0)
                    / 16) as f32
                    * CHUNK,
                grid_min_z,
                0.0,
            ];

            let pc = VisibilityPushConstants {
                view_proj: view_proj.to_cols_array_2d(),
                grid_origin_ws,
                radius: 32,
                height: vb.height,
                _padding: [0, 0],
            };
            if state.render_aabbs {
                ctx.cmd_begin_debug_label(cmd, "Draw AABBs");
                self.render_aabbs(ctx, cmd, &pc, frame_index);
                ctx.cmd_end_debug_label(cmd);
            }

            Some(pc)
        } else {
            None
        };

        self.render_targets.end(ctx.device(), cmd);
        ctx.cmd_end_debug_label(cmd);

        ctx.cmd_begin_debug_label(cmd, "HiZ Pyramid Generation");
        self.hiz_compute.dispatch_all_levels(
            ctx,
            cmd,
            image_index,
            &self.render_targets.depth_pyramids[image_index as usize],
            &self.render_targets.depth_images[image_index as usize],
            extent.width,
            extent.height,
        );
        ctx.cmd_end_debug_label(cmd);

        if let Some(vb) = &mut self.visibility_buffers {
            ctx.cmd_begin_debug_label(cmd, "Visibility Compute");
            self.visibility_compute.dispatch(
                ctx,
                cmd,
                image_index as usize,
                frame_index,
                vb,
                &visibility_push_constants.unwrap(),
            );

            unsafe {
                ctx.device().cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::VERTEX_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .buffer(vb.outputs[frame_index].buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)],
                    &[],
                );
            }

            ctx.cmd_end_debug_label(cmd);
        };

        ctx.cmd_end_debug_label(cmd);
    }

    pub fn draw(
        &mut self,
        ctx: &VkContext,
        cmd: vk::CommandBuffer,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        frame_index: usize,
        wireframe_mode: bool,
    ) {
        let device = ctx.device();
        let push = PushConstants { view_proj };

        ctx.cmd_begin_debug_label(cmd, "Draw Blocks");
        let current_pipeline = self.pipelines.block_pipeline(wireframe_mode);

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, current_pipeline);

            device.cmd_push_constants(
                cmd,
                self.pipelines.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    &push as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>(),
                ),
            );

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                0,
                &[self.descriptors.set],
                &[],
            );
        }

        for (pos, mesh) in &self.mesh_store.blocks {
            let pos_min = Vec3::new(
                pos.x as f32 * 16.0,
                pos.y as f32 * 16.0,
                pos.z as f32 * 16.0,
            );
            let pos_max = Vec3::new(pos_min.x + 16.0, pos_min.y + 16.0, pos_min.z + 16.0);

            if !visibility::aabb_visible(view_proj, pos_min, pos_max) {
                continue;
            }

            let vertex_buffers = [mesh.buffer.buffer];
            let offsets = [mesh.vertex_offset];
            unsafe {
                device.cmd_bind_vertex_buffers(cmd, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    cmd,
                    mesh.buffer.buffer,
                    mesh.index_offset,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
            }
        }
        ctx.cmd_end_debug_label(cmd);

        ctx.cmd_begin_debug_label(cmd, "Draw Water");
        let water_pipeline = self.pipelines.water_pipeline(wireframe_mode);

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, water_pipeline);

            device.cmd_push_constants(
                cmd,
                self.pipelines.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    &push as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>(),
                ),
            );

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                0,
                &[self.descriptors.set],
                &[],
            );
        }

        let mut water_meshes: Vec<_> = self.mesh_store.water.iter().collect();
        water_meshes.sort_by(|(a, _), (b, _)| {
            let dist = |pos: &ChunkSectionPos| {
                camera_pos.distance_squared(glam::Vec3::new(
                    pos.x as f32 * 16.0 + 8.0,
                    pos.y as f32 * 16.0 + 8.0,
                    pos.z as f32 * 16.0 + 8.0,
                ))
            };

            dist(a).partial_cmp(&dist(b)).unwrap_or(Ordering::Equal)
        });

        for (_, mesh) in water_meshes {
            let vertex_buffers = [mesh.buffer.buffer];
            let offsets = [mesh.vertex_offset];
            unsafe {
                device.cmd_bind_vertex_buffers(cmd, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    cmd,
                    mesh.buffer.buffer,
                    mesh.index_offset,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
            }
        }
        ctx.cmd_end_debug_label(cmd);
    }

    pub fn upload_dirty_textures(
        &mut self,
        ctx: &VkContext,
        cmd: vk::CommandBuffer,
        frame_index: usize,
    ) {
        let dirty = self.animation_manager.dirty_textures(&self.assets.textures);
        if dirty.is_empty() {
            return;
        }

        let mut buffer_data = Vec::new();
        let mut regions = Vec::new();

        for (name, tex, (fw, fh), frame_idx) in dirty {
            if let Some(placed) = self.assets.block_atlas.sprites.get(name) {
                let (fx, fy) = tex
                    .animation
                    .as_ref()
                    .unwrap()
                    .get_frame(frame_idx, tex.size());

                let frame_img = tex.data.view(fx, fy, fw, fh).to_image();
                let bytes = frame_img.as_raw();

                let offset = buffer_data.len() as vk::DeviceSize;
                buffer_data.extend_from_slice(bytes);

                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(0)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .image_offset(vk::Offset3D {
                            x: placed.x as i32,
                            y: placed.y as i32,
                            z: 0,
                        })
                        .image_extent(vk::Extent3D {
                            width: fw,
                            height: fh,
                            depth: 1,
                        }),
                );
            }
        }

        let needed_size = buffer_data.len() as vk::DeviceSize;

        let mut staging = Buffer::new(
            ctx,
            needed_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::Auto,
            true,
        );
        staging.upload_data(ctx, 0, &buffer_data);

        unsafe {
            let subresource = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            ctx.device().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(self.blocks_texture.image)
                    .subresource_range(subresource)],
            );

            ctx.device().cmd_copy_buffer_to_image(
                cmd,
                staging.buffer,
                self.blocks_texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );

            ctx.device().cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(self.blocks_texture.image)
                    .subresource_range(subresource)],
            );
        }

        self.staging.push(frame_index, staging);
    }

    pub fn render_aabbs(
        &self,
        ctx: &VkContext,
        cmd: vk::CommandBuffer,
        push_constants: &VisibilityPushConstants,
        frame_index: usize,
    ) {
        let side = (push_constants.radius * 2 + 1) as u32;
        let instance_count = side * side * push_constants.height as u32;

        self.aabb_renderer.draw(
            ctx.device(),
            cmd,
            push_constants,
            instance_count,
            frame_index,
        );
    }

    pub fn recreate_swapchain(&mut self, ctx: &VkContext, swapchain: &Swapchain) {
        self.render_targets.recreate(ctx, swapchain);
        self.hiz_compute.recreate(
            ctx,
            &self.render_targets.depth_pyramids,
            &self.render_targets.depth_images,
        );
        self.visibility_compute
            .recreate_image_sets(ctx, &self.render_targets.depth_pyramids);
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        let device = ctx.device();

        self.mesh_store.drain_and_destroy(ctx);
        self.staging.destroy_all(ctx);

        self.hiz_compute.destroy(ctx);
        self.render_targets.destroy(ctx);
        self.blocks_texture.destroy(ctx);
        if let Some(mut vb) = self.visibility_buffers.take() {
            vb.destroy(ctx);
        }
        self.visibility_compute.destroy(ctx);
        self.aabb_renderer.destroy(device);

        self.pipelines.destroy(device);
        self.descriptors.destroy(device);
    }
}

fn calc_dirty_size(textures: &HashMap<String, TextureEntry>, dirty: &[&str]) -> vk::DeviceSize {
    dirty
        .iter()
        .filter_map(|name| textures.get(*name))
        .map(|tex| {
            let (fw, fh) = tex.size();
            (fw * fh * 4) as vk::DeviceSize
        })
        .sum()
}
