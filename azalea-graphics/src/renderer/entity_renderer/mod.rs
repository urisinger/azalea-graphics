use std::{collections::HashMap, sync::Arc};

use ash::vk;
use azalea_assets::{Assets, entity::ModelPart};
use glam::{Mat4, Vec2, Vec3};
use vk_mem::MemoryUsage;

use crate::renderer::{
    entity_renderer::state::RenderState,
    frame_ctx::FrameCtx,
    vulkan::{buffer::Buffer, context::VkContext},
};

mod renderers;
mod state;
mod pipelines;

struct EntityVertex {
    pub pos: Vec3,
    pub transform_id: u32,
    pub uv: Vec2,
}

struct EntityModel {
    pub offset: u32,
    pub size: u32,
}

pub struct EntityRenderer {
    assets: Arc<Assets>,

    //entity_pipeline: vk::Pipeline,
    //entity_pipeline_layout: vk::PipelineLayout,

    model_vertices: Buffer,
    loaded_models: HashMap<String, EntityModel>,
}

impl EntityRenderer {
    pub fn new(ctx: &VkContext, assets: Arc<Assets>) -> Self {
        let mut buf = Vec::new();
        let loaded_models = assets
            .entity_models
            .iter()
            .map(|(name, model)| {
                let offset = buf.len() as u32;
                let size = Self::load_model_part(&mut buf, 0, model);

                (name.clone(), EntityModel { offset, size })
            })
            .collect();

        let mut staging = Buffer::new(
            ctx,
            buf.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            true,
        );
        let cmd = ctx.begin_one_time_commands();

        staging.upload_data(ctx, 0, &buf);
        let model_vertices = Buffer::new(
            ctx,
            buf.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryUsage::AutoPreferDevice,
            false,
        );
        ctx.end_one_time_commands(cmd);
        Self {
            assets,
            loaded_models,
            model_vertices,
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

    fn render_model(&self, frame_ctx: FrameCtx, model: EntityModel, transform: Mat4) {
        let device = frame_ctx.ctx.device();
        unsafe { device.cmd_draw(frame_ctx.cmd, model.size, 1, model.offset, 0) };
    }

    pub fn render(&mut self, ctx: FrameCtx, states: Vec<RenderState>) {

        let device = ctx.ctx.device();
        unsafe { device.cmd_bind_vertex_buffers(ctx.cmd, 0, &[self.model_vertices.buffer], &[0]) };
        for state in states {
            match state {
                RenderState::Zombie(s) => {
                    let entity_model = s.parent.parent.parent.parent;
                    let zombie_model = &self.loaded_models["zombie#main"];

                }
            }
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
