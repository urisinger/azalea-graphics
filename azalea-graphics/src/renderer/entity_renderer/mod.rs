use std::{collections::HashMap, sync::Arc};

use ash::vk;
use azalea_assets::Assets;
use glam::{Mat4, Vec3};
use parking_lot::Mutex;
use vk_mem::MemoryUsage;

use self::{
    models::zombie::ZombieModel,
    pipelines::create_entity_pipeline,
    state::RenderState,
    transform::ModelTransforms,
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

mod models;
mod pipelines;
mod render_pass;
mod renderers;
pub mod state;
mod transform;
mod types;

#[derive(Clone, Copy)]
struct EntityModel {
    offset: u32,
    size: u32,
}

pub struct EntityRenderer {
    assets: Arc<Assets>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    entity_pipeline: vk::Pipeline,
    entity_pipeline_layout: vk::PipelineLayout,
    loaded_models: HashMap<String, EntityModel>,

    model_vertices: Buffer,
    transform_buffers: [Buffer; MAX_FRAMES_IN_FLIGHT],

    world_descriptor_layout: vk::DescriptorSetLayout,
    world_descriptor_pool: vk::DescriptorPool,
    world_descriptor_sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],

    entities: Arc<Mutex<Vec<RenderState>>>,
}

struct PendingDraw {
    vertex_offset: u32,
    vertex_count: u32,
    transform_offset: u32,
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
                let start = buf.len();
                buf.extend(model.vertices.iter().map(|v| EntityVertex{
                    pos: v.pos,
                    uv: v.uv,
                    transform_id: v.transform_id
                }));
                let end = buf.len();
                (
                    name.clone(),
                    EntityModel {
                        offset: start as u32,
                        size: (end - start) as u32,
                    },
                )
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

        // Create transform buffers (storage buffers for entity transforms)
        const MAX_TRANSFORMS: usize = 1024;
        let transform_buffers: [Buffer; MAX_FRAMES_IN_FLIGHT] = std::array::from_fn(|_| {
            Buffer::new(
                ctx,
                (MAX_TRANSFORMS * size_of::<Mat4>()) as vk::DeviceSize,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryUsage::AutoPreferDevice,
                false,
            )
        });

        // Descriptor set layout for world uniforms and transforms
        let world_descriptor_layout = unsafe {
            ctx.device()
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::VERTEX),
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::VERTEX),
                    ]),
                    None,
                )
                .unwrap()
        };

        let world_descriptor_pool = unsafe {
            ctx.device()
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_BUFFER,
                                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
                            },
                        ]),
                    None,
                )
                .unwrap()
        };

        let layouts = [world_descriptor_layout; MAX_FRAMES_IN_FLIGHT];
        let world_descriptor_sets: [_; _] = unsafe {
            ctx.device()
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(world_descriptor_pool)
                        .set_layouts(&layouts),
                )
                .unwrap()
                .try_into()
                .unwrap()
        };

        // Update descriptor sets with uniform buffers and transform buffers
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                ctx.device().update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                offset: 0,
                                range: size_of::<Uniform>() as u64,
                                buffer: uniforms[i].buffer,
                            }])
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .dst_set(world_descriptor_sets[i])
                            .dst_binding(0),
                        vk::WriteDescriptorSet::default()
                            .buffer_info(&[vk::DescriptorBufferInfo {
                                offset: 0,
                                range: vk::WHOLE_SIZE,
                                buffer: transform_buffers[i].buffer,
                            }])
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .dst_set(world_descriptor_sets[i])
                            .dst_binding(1),
                    ],
                    &[],
                )
            };
        }

        let (entity_pipeline_layout, entity_pipeline) = create_entity_pipeline(
            ctx,
            module,
            world_descriptor_layout,
            texture_manager.descriptor_set_layout(),
            render_pass,
        );

        Self {
            assets,
            world_descriptor_layout,
            world_descriptor_pool,
            world_descriptor_sets,
            loaded_models,
            render_pass,
            framebuffers,
            model_vertices,
            transform_buffers,
            entity_pipeline,
            entity_pipeline_layout,
            entities,
        }
    }

    fn render_model(&self, frame_ctx: &mut FrameCtx, draw: &PendingDraw) {
        let device = frame_ctx.ctx.device();

        let push_constants = EntityPushConstants {
            tex_id: draw.texture,
            transform_offset: draw.transform_offset,
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
            device.cmd_draw(frame_ctx.cmd, draw.vertex_count, 1, draw.vertex_offset, 0)
        };
    }

    pub fn render(&mut self, frame_ctx: &mut FrameCtx, texture_manager: &mut TextureManager) {
        let states = self.entities.lock();
        if states.is_empty() {
            return;
        }

        // Collect all transforms and prepare draw calls
        let mut all_transforms = Vec::new();
        let mut pending: Vec<PendingDraw> = Vec::new();

        let zombie_model_data = self
            .assets
            .entity_models
            .get("minecraft:zombie#main")
            .expect("Zombie model not found");
        let zombie_model = ZombieModel::new(zombie_model_data);

        for state in states.iter() {
            match state {
                RenderState::Zombie(s) => {
                    let transform_offset = all_transforms.len() as u32;

                    // Create transforms and animate
                    let mut model_transforms = ModelTransforms::new(zombie_model_data);
                    zombie_model.set_angles(&mut model_transforms, s);

                    // Build world transform following Minecraft's matrix stack operations:
                    // matrixStack.scale(baseScale, baseScale, baseScale)
                    // setupTransforms() -> rotation by bodyYaw
                    // matrixStack.scale(-1, -1, 1)
                    // matrixStack.translate(0, -1.501, 0)
                    
                    // Start with world position
                    let mut world_transform = Mat4::from_scale(Vec3::splat(s.base_scale));
                    

                    world_transform *= Mat4::from_translation(Vec3::new(s.x as f32, s.y as f32, s.z as f32));

                    // Convert to Mat4 array and add to buffer
                    let transforms =
                        model_transforms.to_transforms(zombie_model_data, world_transform);
                    all_transforms.extend(transforms);

                    let texture =
                        texture_manager.get_texture(frame_ctx, "textures/entity/zombie/zombie.png");
                    let model = self.loaded_models["minecraft:zombie#main"];

                    pending.push(PendingDraw {
                        vertex_offset: model.offset,
                        vertex_count: model.size,
                        transform_offset,
                        texture,
                    });
                }
            }
        }

        drop(states); // Release lock

        if pending.is_empty() {
            return;
        }

        // Upload transforms to GPU
        frame_ctx.upload_to(
            &all_transforms,
            &self.transform_buffers[frame_ctx.frame_index],
        );

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
                    self.world_descriptor_sets[frame_ctx.frame_index],
                    texture_manager.get_descriptor_set(device, frame_ctx.frame_index),
                ],
                &[],
            );
        }

        // Render all entities
        for draw in pending.iter() {
            self.render_model(frame_ctx, draw);
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
                .destroy_descriptor_set_layout(self.world_descriptor_layout, None);
            ctx.device()
                .destroy_descriptor_pool(self.world_descriptor_pool, None);
        }
        self.model_vertices.destroy(ctx);
        for buffer in &mut self.transform_buffers {
            buffer.destroy(ctx);
        }
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
