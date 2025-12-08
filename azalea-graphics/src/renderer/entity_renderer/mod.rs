use std::{collections::HashMap, sync::Arc};

use ash::vk;
use azalea_assets::{Assets, entity::ModelPart};
use glam::{Mat4, Vec2, Vec3};
use parking_lot::Mutex;
use vk_mem::MemoryUsage;

use self::{
    pipelines::create_entity_pipeline,
    state::RenderState,
    types::{EntityPushConstants, EntityVertex},
};
use crate::renderer::{
    Uniform,
    entity_renderer::render_pass::create_entity_render_pass,
    frame_ctx::FrameCtx,
    render_targets::RenderTargets,
    texture_manager::TextureManager,
    utils::create_framebuffers,
    vulkan::{buffer::Buffer, context::VkContext, frame_sync::MAX_FRAMES_IN_FLIGHT},
};

mod pipelines;
mod render_pass;
mod renderers;
pub mod state;
mod types;

#[derive(Debug, Clone, Copy)]
struct EntityModel {
    pub offset: u32,
    pub size: u32,
}

pub struct EntityRenderer {
    assets: Arc<Assets>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    entity_pipeline: vk::Pipeline,
    entity_pipeline_layout: vk::PipelineLayout,

    model_vertices: Buffer,
    loaded_models: HashMap<String, EntityModel>,

    uniform_descriptor_layout: vk::DescriptorSetLayout,
    uniform_descriptor_pool: vk::DescriptorPool,
    uniform_descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],

    entities: Arc<Mutex<Vec<RenderState>>>,
}

struct PendingDraw {
    model: EntityModel,
    transform: Mat4,
    texture: u32,
}

impl EntityRenderer {
    pub fn new(
        ctx: &VkContext,
        module: vk::ShaderModule,
        assets: Arc<Assets>,
        render_targets: &RenderTargets,
        texture_manager: &TextureManager,
        entities: Arc<Mutex<Vec<RenderState>>>,
        uniforms: &[Buffer; MAX_FRAMES_IN_FLIGHT],
    ) -> Self {
        let mut buf = Vec::new();
        let loaded_models = assets
            .entity_models
            .iter()
            .map(|(name, model)| {
                let offset = (buf.len()) as u32;
                let size = Self::load_model_part(&mut buf, 0, model);

                (name.clone(), EntityModel { offset, size })
            })
            .collect();

        let mut staging = Buffer::new_staging(
            ctx,
            (buf.len() * size_of::<EntityVertex>()) as vk::DeviceSize,
        );
        let cmd = ctx.begin_one_time_commands();

        staging.upload_data(ctx, 0, &buf);
        let model_vertices = Buffer::new(
            ctx,
            (buf.len() * size_of::<EntityVertex>()) as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryUsage::AutoPreferDevice,
            false,
        );
        staging.copy_to(ctx, &model_vertices, cmd);

        ctx.end_one_time_commands(cmd);

        staging.destroy(ctx);

        let render_pass = create_entity_render_pass(ctx, render_targets);
        let framebuffers = create_framebuffers(ctx, render_targets, render_pass);

        let uniform_descriptor_layout = unsafe {
            ctx.device()
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::VERTEX),
                    ]),
                    None,
                )
                .unwrap()
        };

        let uniform_descriptor_pool = unsafe {
            ctx.device()
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
                        .pool_sizes(&[vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
                        }]),
                    None,
                )
                .unwrap()
        };

        let layouts = [uniform_descriptor_layout; MAX_FRAMES_IN_FLIGHT];
        let uniform_descriptor_sets: [_; _] = unsafe {
            ctx.device()
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(uniform_descriptor_pool)
                        .set_layouts(&layouts),
                )
                .unwrap()
                .try_into()
                .unwrap()
        };

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                ctx.device().update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            offset: 0,
                            range: size_of::<Uniform>() as u64,
                            buffer: uniforms[i].buffer,
                        }])
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .dst_set(uniform_descriptor_sets[i])],
                    &[],
                )
            };
        }

        let (entity_pipeline_layout, entity_pipeline) = create_entity_pipeline(
            ctx,
            module,
            uniform_descriptor_layout,
            texture_manager.descriptor_set_layout(),
            render_pass,
        );

        Self {
            assets,
            uniform_descriptor_layout,
            uniform_descriptor_pool,
            uniform_descriptor_sets,
            render_pass,
            framebuffers,
            loaded_models,
            model_vertices,
            entity_pipeline,
            entity_pipeline_layout,
            entities,
        }
    }

    fn load_model_part(
        vertices: &mut Vec<EntityVertex>,
        transform_id: u32,
        model_part: &ModelPart,
    ) -> u32 {
        let mut num_vertices = 0;
        for cuboid in &model_part.cuboids {
            for side in &cuboid.sides {
                for vertex in &side.vertices {
                    vertices.push(EntityVertex {
                        pos: vertex.pos,
                        transform_id,
                        uv: vertex.uv.unwrap_or(Vec2::ZERO),
                    });

                    num_vertices += 1;
                }
            }
        }
        for (_, child) in &model_part.children {
            num_vertices += Self::load_model_part(vertices, transform_id + 1, &child);
        }
        num_vertices
    }
    fn render_model(
        &self,
        frame_ctx: &mut FrameCtx,
        model: &EntityModel,
        transform: Mat4,
        tex_id: u32,
    ) {
        let device = frame_ctx.ctx.device();

        let push_constants = EntityPushConstants {
            model: transform,
            tex_id,
        };
        unsafe {
            device.cmd_push_constants(
                frame_ctx.cmd,
                self.entity_pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const _ as *const u8,
                    std::mem::size_of::<EntityPushConstants>(),
                ),
            );
            device.cmd_draw(frame_ctx.cmd, model.size, 1, model.offset, 0)
        };
    }

    pub fn render(&mut self, frame_ctx: &mut FrameCtx, texture_manager: &mut TextureManager) {
        let mut pending = Vec::new();
        {
            let states = self.entities.lock();
            for state in states.iter() {
                match state {
                    RenderState::Zombie(s) => {
                        let entity_model = &s.parent.parent.parent.parent;
                        let zombie_model = &self.loaded_models["minecraft:zombie#main"];
                        let transform = Mat4::from_translation(Vec3::new(
                            entity_model.x as f32,
                            entity_model.y as f32,
                            entity_model.z as f32,
                        ));

                        pending.push(PendingDraw {
                            model: *zombie_model,
                            transform,
                            texture: texture_manager
                                .get_texture(frame_ctx, "textures/entity/zombie/zombie.png"),
                        });
                    }
                }
            }
        }

        if pending.is_empty() {
            return;
        }

        let device = frame_ctx.ctx.device();
        self.begin(frame_ctx);

        unsafe {
            device.cmd_bind_vertex_buffers(frame_ctx.cmd, 0, &[self.model_vertices.buffer], &[0]);
            device.cmd_bind_pipeline(
                frame_ctx.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.entity_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                frame_ctx.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.entity_pipeline_layout,
                0,
                &[
                    self.uniform_descriptor_sets[frame_ctx.frame_index],
                    texture_manager.get_descriptor_set(device, frame_ctx.frame_index),
                ],
                &[],
            );
        }

        // Replay draws that we collected
        for draw in pending.iter() {
            self.render_model(frame_ctx, &draw.model, draw.transform, draw.texture);
        }

        self.end(frame_ctx);
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

    pub fn recreate_swapchain(&mut self, ctx: &VkContext, render_targets: &RenderTargets) {
        for fb in self.framebuffers.drain(..) {
            unsafe {
                ctx.device().destroy_framebuffer(fb, None);
            }
        }

        self.framebuffers = create_framebuffers(ctx, render_targets, self.render_pass);
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        unsafe { ctx.device().destroy_render_pass(self.render_pass, None) };
        for framebuffer in self.framebuffers.drain(..) {
            unsafe {
                ctx.device().destroy_framebuffer(framebuffer, None);
            }
        }
        unsafe {
            ctx.device()
                .destroy_pipeline_layout(self.entity_pipeline_layout, None);
            ctx.device().destroy_pipeline(self.entity_pipeline, None);
            ctx.device()
                .destroy_descriptor_set_layout(self.uniform_descriptor_layout, None);
            ctx.device()
                .destroy_descriptor_pool(self.uniform_descriptor_pool, None);
        }
        self.model_vertices.destroy(ctx);
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityPose {
    Standing = 0,
    Gliding = 1,
    Sleeping = 2,
    Swimming = 3,
    SpinAttack = 4,
    Crouching = 5,
    LongJumping = 6,
    Dying = 7,
    Croaking = 8,
    UsingTongue = 9,
    Sitting = 10,
    Roaring = 11,
    Sniffing = 12,
    Emerging = 13,
    Digging = 14,
    Sliding = 15,
    Shooting = 16,
    Inhaling = 17,
}

impl EntityPose {
    pub fn index(self) -> u8 {
        self as u8
    }

    pub fn from_index(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Standing),
            1 => Some(Self::Gliding),
            2 => Some(Self::Sleeping),
            3 => Some(Self::Swimming),
            4 => Some(Self::SpinAttack),
            5 => Some(Self::Crouching),
            6 => Some(Self::LongJumping),
            7 => Some(Self::Dying),
            8 => Some(Self::Croaking),
            9 => Some(Self::UsingTongue),
            10 => Some(Self::Sitting),
            11 => Some(Self::Roaring),
            12 => Some(Self::Sniffing),
            13 => Some(Self::Emerging),
            14 => Some(Self::Digging),
            15 => Some(Self::Sliding),
            16 => Some(Self::Shooting),
            17 => Some(Self::Inhaling),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            EntityPose::Standing => "standing",
            EntityPose::Gliding => "fall_flying",
            EntityPose::Sleeping => "sleeping",
            EntityPose::Swimming => "swimming",
            EntityPose::SpinAttack => "spin_attack",
            EntityPose::Crouching => "crouching",
            EntityPose::LongJumping => "long_jumping",
            EntityPose::Dying => "dying",
            EntityPose::Croaking => "croaking",
            EntityPose::UsingTongue => "using_tongue",
            EntityPose::Sitting => "sitting",
            EntityPose::Roaring => "roaring",
            EntityPose::Sniffing => "sniffing",
            EntityPose::Emerging => "emerging",
            EntityPose::Digging => "digging",
            EntityPose::Sliding => "sliding",
            EntityPose::Shooting => "shooting",
            EntityPose::Inhaling => "inhaling",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ArmPose {
    Empty,
    Item,
    Block,
    BowAndArrow,
    ThrowSpear,
    CrossbowHold,
    Spyglass,
    TootHorn,
    Brush,
}
