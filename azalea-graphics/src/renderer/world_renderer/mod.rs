use std::{array::from_fn, cmp::Ordering, collections::HashMap, sync::Arc};

use ash::vk;
use azalea::core::position::ChunkSectionPos;
use azalea_assets::{Assets, processed::atlas::TextureEntry};
use glam::{Vec3, Vec4};
use image::GenericImageView;
use vk_mem::MemoryUsage;

use crate::{
    app::WorldUpdate,
    renderer::{
        frame_ctx::FrameCtx, hiz, render_targets::RenderTargets, timings, utils::create_framebuffers, vulkan::{
            buffer::Buffer,
            context::VkContext,
            frame_sync::{FrameSync, MAX_FRAMES_IN_FLIGHT},
            texture::Texture,
        }, world_renderer::{
            aabb_renderer::AabbRenderer,
            animation::AnimationManager,
            mesher::Mesher,
            render_pass::create_world_render_pass,
            types::{VisibilityUniform},
            visibility::{buffers::VisibilityBuffers, compute::VisibilityCompute},
        }
    },
};

mod aabb_renderer;
mod animation;
mod descriptors;
mod mesher;
mod meshes;
mod pipelines;
mod render_pass;
mod types;
mod visibility;

use descriptors::Descriptors;
use meshes::MeshStore;
use pipelines::{PipelineOptions, Pipelines};
use types::BlockVertex;

pub struct WorldRenderer {
    mesher: Option<Mesher>,

    animation_manager: AnimationManager,
    mesh_store: MeshStore,

    hiz_compute: hiz::HiZCompute,
    visibility_compute: VisibilityCompute,
    visibility_buffers: Option<VisibilityBuffers>,
    aabb_renderer: AabbRenderer,

    visibility_uniforms: [Buffer; MAX_FRAMES_IN_FLIGHT],

    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    pipelines: Pipelines,
    descriptors: Descriptors,
    blocks_texture: Texture,
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
    pub disable_visibilty: bool,
    pub render_distance: u32,
    pub worker_threads: u32,
}

impl Default for WorldRendererConfig {
    fn default() -> Self {
        Self {
            wireframe_mode: false,
            render_aabbs: false,
            disable_visibilty: false,
            render_distance: 32,
            worker_threads: num_cpus::get() as u32 / 2,
        }
    }
}

impl WorldRenderer {
    pub fn new(
        assets: Arc<Assets>,
        ctx: &VkContext,
        module: vk::ShaderModule,
        render_targets: &RenderTargets,
        uniforms: &[Buffer; MAX_FRAMES_IN_FLIGHT],
        options: WorldRendererFeatures,
    ) -> Self {
        let atlas_image =
            animation::create_initial_atlas(&assets.block_atlas, &assets.block_textures);
        let blocks_texture = Texture::from_image(ctx, atlas_image);



        let render_pass = create_world_render_pass(ctx, render_targets);
        let framebuffers = create_framebuffers(ctx, render_targets, render_pass);

        let descriptors = Descriptors::new(ctx.device(), &uniforms, &blocks_texture);

        let pipelines = Pipelines::new(
            ctx,
            render_pass,
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

        let visibility_uniforms: [_; MAX_FRAMES_IN_FLIGHT] = from_fn(|i| {
            Buffer::new(
                ctx,
                size_of::<VisibilityUniform>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryUsage::AutoPreferDevice,
                false,
            )
        });

        let visibility_compute = VisibilityCompute::new(
            ctx,
            module,
            &visibility_uniforms,
            &render_targets.depth_pyramids,
            32,
            1,
        );
        let aabb_renderer = AabbRenderer::new(ctx, &visibility_uniforms, module, render_pass);

        Self {
            mesher: None,
            animation_manager: AnimationManager::from_textures(&assets.block_textures),
            hiz_compute,

            visibility_uniforms,

            visibility_compute,
            visibility_buffers: None,
            aabb_renderer,
            render_pass,
            framebuffers,

            mesh_store: Default::default(),
            pipelines,
            descriptors,
            blocks_texture,
            assets: assets.clone(),
        }
    }

    pub fn tick(&mut self) {
        self.animation_manager.tick(&self.assets.block_textures);
    }

    pub fn update_visibility(&mut self, ctx: &VkContext, frame_index: usize, camera_pos: Vec3) {
        if let (Some(mesher), Some(vis_bufs)) = (&self.mesher, &mut self.visibility_buffers) {
            let cx = (camera_pos.x / 16.0).floor() as i32;
            let cy = (camera_pos.y / 16.0).floor() as i32;
            let cz = (camera_pos.z / 16.0).floor() as i32;
            let min_y = self.mesher.as_ref().unwrap().world.read().chunks.min_y;
            let snapshot = vis_bufs.snapshot(ctx, frame_index, cx, cz, min_y);

            mesher.update_visibility(snapshot);
        }
    }

    pub fn average_mesh_time_ms(&self) -> f32 {
        if let Some(mesher) = &self.mesher {
            mesher.average_mesh_time_ms()
        } else {
            0.0
        }
    }

    pub fn update(
        &mut self,
        ctx: &VkContext,
        config: &WorldRendererConfig,
        update: WorldUpdate,
        sync: &mut FrameSync,
    ) {
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
                unsafe { ctx.device().queue_wait_idle(ctx.graphics_queue()).unwrap() };
                let world_read = world.read();
                let max_height = world_read.chunks.height as i32 - world_read.chunks.min_y;
                drop(world_read);

                let radius = config.render_distance as i32;
                let height = max_height / 16;

                if let Some(vb) = &mut self.visibility_buffers {
                    vb.recreate(ctx, radius, height);
                } else {
                    let vb = VisibilityBuffers::new(ctx, radius, height);
                    self.visibility_buffers = Some(vb);
                }

                let vb = self.visibility_buffers.as_ref().unwrap();

                for f in 0..MAX_FRAMES_IN_FLIGHT {
                    self.visibility_compute
                        .rewrite_frame_set(ctx.device(), f, &vb.outputs[f]);
                }

                self.aabb_renderer
                    .recreate_descriptor_sets(ctx.device(), &vb.outputs);

                self.mesher = Some(Mesher::new(self.assets.clone(), world));
            }
        }
    }

    pub fn set_render_distance(&mut self, ctx: &VkContext, new_distance: u32) {
        if let Some(mesher) = &self.mesher {
            let world_read = mesher.world.read();
            let max_height = world_read.chunks.height as i32 - world_read.chunks.min_y;
            drop(world_read);

            let radius = new_distance as i32;
            let height = max_height / 16;

            if let Some(vb) = &mut self.visibility_buffers {
                if vb.radius != radius || vb.height != height {
                    unsafe { ctx.device().queue_wait_idle(ctx.graphics_queue()).unwrap() };
                    vb.recreate(ctx, radius, height);

                    for f in 0..MAX_FRAMES_IN_FLIGHT {
                        self.visibility_compute
                            .rewrite_frame_set(ctx.device(), f, &vb.outputs[f]);
                    }

                    self.aabb_renderer
                        .recreate_descriptor_sets(ctx.device(), &vb.outputs);
                }
            }
        }
    }

    pub fn set_worker_threads(&mut self, ctx: &VkContext, new_thread_count: u32) {
        if let Some(mesher) = &mut self.mesher {
            mesher.set_worker_threads(new_thread_count);
        }
    }

    pub fn render(&mut self, frame_ctx: &mut FrameCtx) {
        let ctx = frame_ctx.ctx;
        let camera_pos = frame_ctx.camera_pos;
        let view_proj = frame_ctx.view_proj;

        if let Some(vb) = &mut self.visibility_buffers {
            const CHUNK: f32 = 16.0;

            let cam_chunk_x = (camera_pos.x / CHUNK).floor() as i32;
            let cam_chunk_z = (camera_pos.z / CHUNK).floor() as i32;
            let grid_min_x = (cam_chunk_x) as f32 * CHUNK;
            let grid_min_z = (cam_chunk_z) as f32 * CHUNK;
            let grid_origin_ws = Vec4::new(
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
            );

            let visibility_uniform = VisibilityUniform {
                view_proj,
                grid_origin_ws,
                radius: frame_ctx.config.render_distance as i32,
                height: vb.height,
            };

            frame_ctx.upload_to(
                &[visibility_uniform],
                &self.visibility_uniforms[frame_ctx.frame_index],
            );
        }

        ctx.cmd_begin_debug_label(
            frame_ctx.cmd,
            &format!("World Render Frame {}", frame_ctx.frame_index),
        );

        ctx.cmd_begin_debug_label(frame_ctx.cmd, "Update meshes");
        self.mesh_store
            .process_mesher_results(frame_ctx, &self.mesher);

        ctx.cmd_end_debug_label(frame_ctx.cmd);

        ctx.cmd_begin_debug_label(frame_ctx.cmd, "Update dirty textures");
        frame_ctx.begin_timestamp(timings::START_UPLOAD_DIRTY);

        self.upload_dirty_textures(frame_ctx);

        ctx.cmd_end_debug_label(frame_ctx.cmd);
        frame_ctx.end_timestamp(timings::END_UPLOAD_DIRTY);

        frame_ctx.begin_timestamp(timings::START_TERRAIN_PASS);
        ctx.cmd_begin_debug_label(frame_ctx.cmd, "Main Render Pass");
        self.begin(frame_ctx);
        self.draw(frame_ctx, camera_pos);

        if let Some(vb) = &mut self.visibility_buffers {
            if frame_ctx.config.render_aabbs {
                ctx.cmd_begin_debug_label(frame_ctx.cmd, "Draw AABBs");
                let side = (frame_ctx.config.render_distance * 2 + 1) as u32;
                let instance_count = side * side * vb.height as u32;
                self.aabb_renderer.draw(
                    ctx.device(),
                    frame_ctx.cmd,
                    instance_count,
                    frame_ctx.frame_index,
                );
                ctx.cmd_end_debug_label(frame_ctx.cmd);
            }
        }

        self.end(frame_ctx);

        ctx.cmd_end_debug_label(frame_ctx.cmd);
        frame_ctx.end_timestamp(timings::END_TERRAIN_PASS);

        frame_ctx.begin_timestamp(timings::START_HIZ_COMPUTE);
        ctx.cmd_begin_debug_label(frame_ctx.cmd, "HiZ Pyramid Generation");
        self.hiz_compute.dispatch_all_levels(frame_ctx);
        ctx.cmd_end_debug_label(frame_ctx.cmd);
        frame_ctx.end_timestamp(timings::END_HIZ_COMPUTE);

        frame_ctx.end_timestamp(timings::START_VISIBILITY_COMPUTE);
        if let Some(vb) = &mut self.visibility_buffers
            && !frame_ctx.config.disable_visibilty
        {
            ctx.cmd_begin_debug_label(frame_ctx.cmd, "Visibility Compute");
            self.visibility_compute.dispatch(frame_ctx, vb);

            unsafe {
                ctx.device().cmd_pipeline_barrier(
                    frame_ctx.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::VERTEX_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .buffer(vb.outputs[frame_ctx.frame_index].buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)],
                    &[],
                );
            }

            ctx.cmd_end_debug_label(frame_ctx.cmd);
        };
        frame_ctx.end_timestamp(timings::END_VISIBILITY_COMPUTE);

        ctx.cmd_end_debug_label(frame_ctx.cmd);
    }

    pub fn begin(&self, frame_ctx: &FrameCtx) {
        let device = frame_ctx.ctx.device();
        let cmd = frame_ctx.cmd;
        let extent = frame_ctx.render_targets.extent();
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
        ];

        let rp_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[frame_ctx.image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values);

        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_info, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );
        }
    }

    pub fn end(&self, frame_ctx: &FrameCtx) {
        unsafe { frame_ctx.ctx.device().cmd_end_render_pass(frame_ctx.cmd) };
    }

    pub fn draw(&mut self, frame_ctx: &mut FrameCtx, camera_pos: glam::Vec3) {
        let FrameCtx {
            ctx,
            cmd,
            frame_index,
            view_proj,
            config,
            ..
        } = frame_ctx;
        let device = ctx.device();

        ctx.cmd_begin_debug_label(*cmd, "Draw Blocks");
        let current_pipeline = self.pipelines.block_pipeline(config.wireframe_mode);

        unsafe {
            device.cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::GRAPHICS, current_pipeline);

            device.cmd_bind_descriptor_sets(
                *cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines.layout,
                0,
                &[self.descriptors.sets[*frame_index]],
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
                device.cmd_bind_vertex_buffers(*cmd, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    *cmd,
                    mesh.buffer.buffer,
                    mesh.index_offset,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(*cmd, mesh.index_count, 1, 0, 0, 0);
            }
        }
        ctx.cmd_end_debug_label(*cmd);

        ctx.cmd_begin_debug_label(*cmd, "Draw Water");
        let water_pipeline = self.pipelines.water_pipeline(config.wireframe_mode);

        unsafe {
            device.cmd_bind_pipeline(*cmd, vk::PipelineBindPoint::GRAPHICS, water_pipeline);
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

        for (pos, mesh) in water_meshes {
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
                device.cmd_bind_vertex_buffers(*cmd, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    *cmd,
                    mesh.buffer.buffer,
                    mesh.index_offset,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(*cmd, mesh.index_count, 1, 0, 0, 0);
            }
        }
        ctx.cmd_end_debug_label(*cmd);
    }

    pub fn upload_dirty_textures(&mut self, frame_ctx: &mut FrameCtx) {
        let dirty = self
            .animation_manager
            .dirty_textures(&self.assets.block_textures);
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

        let subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        frame_ctx.pipeline_barrier(
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            &[],
            &[vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(self.blocks_texture.image)
                .subresource_range(subresource)],
        );

        frame_ctx.upload_to_image(
            &buffer_data,
            self.blocks_texture.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );

        frame_ctx.pipeline_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
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

    pub fn recreate_swapchain(&mut self, ctx: &VkContext, render_targets: &RenderTargets) {
        for fb in self.framebuffers.drain(..) {
            unsafe { ctx.device().destroy_framebuffer(fb, None) };
        }
        self.framebuffers = create_framebuffers(ctx, render_targets, self.render_pass);

        self.hiz_compute.recreate(
            ctx,
            &render_targets.depth_pyramids,
            &render_targets.depth_images,
        );
        self.visibility_compute
            .recreate_image_sets(ctx, &render_targets.depth_pyramids);
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        let device = ctx.device();

        self.mesh_store.drain_and_destroy(ctx);

        unsafe {
            device.destroy_render_pass(self.render_pass, None);
        }
        for fb in self.framebuffers.drain(..) {
            unsafe { device.destroy_framebuffer(fb, None) };
        }
        self.hiz_compute.destroy(ctx);
        self.blocks_texture.destroy(ctx);

        if let Some(mut vb) = self.visibility_buffers.take() {
            vb.destroy(ctx);
        }
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            self.visibility_uniforms[i].destroy(ctx);
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
