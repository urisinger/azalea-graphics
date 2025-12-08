use std::{array::from_fn, io::Cursor, sync::Arc, time::Duration};

use ash::{util::read_spv, vk};
use crossbeam::channel::Receiver;
pub use entity_renderer::state::RenderState;
use parking_lot::Mutex;
use raw_window_handle::{DisplayHandle, WindowHandle};
use vk_mem::MemoryUsage;
use vulkan::{
    context::VkContext,
    frame_sync::{FrameSync, MAX_FRAMES_IN_FLIGHT},
};
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::KeyCode,
    window::Window,
};

use self::{
    camera::{Camera, CameraController, Projection},
    ui::EguiVulkan,
    world_renderer::{WorldRenderer, WorldRendererFeatures},
};
use crate::{
    app::{RendererArgs, WorldUpdate},
    renderer::{
        entity_renderer::EntityRenderer,
        frame_ctx::FrameCtx,
        render_targets::RenderTargets,
        texture_manager::TextureManager,
        timings::Timings,
        vulkan::{buffer::Buffer, timestamp::TimestampQueryPool},
        world_renderer::WorldRendererConfig,
    },
};

mod camera;
pub mod chunk;
mod entity_renderer;
mod frame_ctx;
mod hiz;
mod mesh;
mod render_targets;
mod texture_manager;
mod timings;
mod ui;
mod utils;
pub mod vulkan;
pub mod world_renderer;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Uniform {
    pub view_proj: glam::Mat4,
}

pub struct Renderer {
    context: VkContext,
    render_targets: RenderTargets,
    should_recreate: bool,
    width: u32,
    height: u32,

    renderer_config: WorldRendererConfig,
    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
    timestamp_pools: Option<[TimestampQueryPool; MAX_FRAMES_IN_FLIGHT]>,

    uniforms: [Buffer; MAX_FRAMES_IN_FLIGHT],

    sync: FrameSync,

    world: WorldRenderer,
    entity_renderer: EntityRenderer,
    texture_manager: TextureManager,

    camera: Camera,
    projection: Projection,
    camera_controller: CameraController,

    egui: EguiVulkan,

    tick_accumulator: Duration,
    tick_interval: Duration,
}

impl Renderer {
    pub fn new(
        window_handle: &WindowHandle,
        display_handle: &DisplayHandle,
        size: PhysicalSize<u32>,
        event_loop: &ActiveEventLoop,
        args: &RendererArgs,
        entities: Arc<Mutex<Vec<RenderState>>>,
    ) -> anyhow::Result<Self> {
        let context = VkContext::new(window_handle, display_handle, args);
        let render_targets = RenderTargets::new(&context, size.width, size.height);

        let max_tex = unsafe {
            let props = context
                .instance()
                .get_physical_device_properties(context.physical_device());
            props.limits.max_image_dimension2_d
        };

        let assets = Arc::new(azalea_assets::load_assets("assets/minecraft", max_tex));

        let texture_manager = TextureManager::new(&context, assets.clone());

        let spirv = read_spv(&mut Cursor::new(include_bytes!(env!("SHADERS")))).unwrap();
        let module = unsafe {
            context
                .device()
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
                .unwrap()
        };
        let uniforms: [_; MAX_FRAMES_IN_FLIGHT] = from_fn(|i| {
            Buffer::new(
                &context,
                size_of::<Uniform>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryUsage::AutoPreferDevice,
                false,
            )
        });

        let entity_renderer = EntityRenderer::new(
            &context,
            module,
            assets.clone(),
            &render_targets,
            &texture_manager,
            entities,
            &uniforms,
        );

        let world = WorldRenderer::new(
            assets.clone(),
            &context,
            module,
            &render_targets,
            &uniforms,
            WorldRendererFeatures {
                fill_mode_non_solid: context.features().fill_mode_non_solid,
            },
        );

        let command_pool = create_command_pool(&context);
        let command_buffers = allocate_command_buffers(&context, command_pool);

        let sync = FrameSync::new(context.device(), render_targets.swapchain.images.len());

        let camera = Camera::new(glam::vec3(0.0, 250.0, 2.0), 0.0, 90.0);
        let projection = Projection::new(size.width, size.height, 90.0, 0.1);
        let camera_controller = CameraController::new(4.0, 1.0);

        let egui = EguiVulkan::new(
            event_loop,
            &context,
            module,
            &render_targets.swapchain,
            None,
        )?;

        let module = unsafe { context.device().destroy_shader_module(module, None) };

        let timestamp_pools = if context.features().timestamp_queries && args.timestamps {
            Some([(); MAX_FRAMES_IN_FLIGHT].map(|_| {
                TimestampQueryPool::new(context.device(), timings::TIMESTAMP_COUNT as u32)
                    .expect("Failed creating timestamp query pool")
            }))
        } else {
            None
        };

        Ok(Self {
            context,
            render_targets,
            should_recreate: false,
            width: size.width,
            height: size.height,
            renderer_config: Default::default(),
            uniforms,

            command_pool,
            command_buffers,
            timestamp_pools,

            sync,
            world,
            camera,
            projection,
            camera_controller,
            entity_renderer,
            texture_manager,

            egui,

            tick_accumulator: Duration::ZERO,
            tick_interval: Duration::from_millis(50),
        })
    }

    pub fn collect_timings(&self, frame: usize) -> Option<Timings> {
        if let Some(timestamps) = &self.timestamp_pools {
            let mut raw_timestamps = [0u64; timings::TIMESTAMP_COUNT];
            timestamps[frame].get_results(self.context.device(), &mut raw_timestamps);

            let properties = unsafe {
                self.context
                    .instance()
                    .get_physical_device_properties(self.context.physical_device())
            };
            let timestamp_period = properties.limits.timestamp_period;

            Some(timings::Timings::from_ticks(
                raw_timestamps,
                timestamp_period,
            ))
        } else {
            None
        }
    }

    pub fn run_debug_ui(&mut self, window: &Window, frame_time_ms: f64) {
        let wireframe_available = self.context.features().fill_mode_non_solid;
        let timings = self.collect_timings(self.sync.current_frame);

        self.egui.run(window, |ctx| {
            egui::Window::new("Debug Info").show(ctx, |ui| {
                ui.label(format!("Frame time: {:.2}ms", frame_time_ms));
                ui.label("Azalea Graphics Renderer");

                ui.separator();

                if let Some(timings) = timings {
                    ui.collapsing("GPU Timings", |ui| {
                        ui.label(format!(
                            "Upload Dirty: {:.2}ms",
                            timings.upload_dirty_time()
                        ));
                        ui.label(format!(
                            "Terrain Pass: {:.2}ms",
                            timings.terrain_pass_time()
                        ));
                        ui.label(format!("HiZ Compute: {:.2}ms", timings.hiz_compute_time()));
                        ui.label(format!(
                            "Visibility Compute: {:.2}ms",
                            timings.visibility_compute_time()
                        ));
                        ui.label(format!("UI Pass: {:.2}ms", timings.ui_time()));
                        ui.label(format!("Total GPU: {:.2}ms", timings.frame_time()));
                    });
                } else {
                    ui.label("GPU timings: Not enabled");
                }

                ui.separator();

                ui.add_enabled(
                    wireframe_available,
                    egui::Checkbox::new(
                        &mut self.renderer_config.wireframe_mode,
                        "Wireframe mode (F3)",
                    ),
                );

                ui.add_enabled(
                    wireframe_available,
                    egui::Checkbox::new(
                        &mut self.renderer_config.render_aabbs,
                        "Render aabbs (F2)",
                    ),
                );
                ui.checkbox(
                    &mut self.renderer_config.disable_visibilty,
                    "Disable visibility calculation (F4)",
                );
                let response = ui.add(
                    egui::Slider::new(&mut self.renderer_config.render_distance, 0..=64)
                        .text("Render distance"),
                );

                if response.changed() {
                    self.world
                        .set_render_distance(&self.context, self.renderer_config.render_distance);
                }
                let worker_threads = self.renderer_config.worker_threads;
                let response = ui.add(
                    egui::Slider::new(
                        &mut self.renderer_config.worker_threads,
                        1..=num_cpus::get() as u32,
                    )
                    .text(format!("Worker threads (current: {})", worker_threads)),
                );

                if response.changed() {
                    self.world
                        .set_worker_threads(&self.context, self.renderer_config.worker_threads);
                }

                ui.label(format!(
                    "Average mesh time: {}ms",
                    self.world.average_mesh_time_ms()
                ))
            });
        });
    }

    pub fn update_world(&mut self, update: WorldUpdate) {
        self.world
            .update(&self.context, &self.renderer_config, update, &mut self.sync);
    }

    pub fn update(&mut self, dt: Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);

        self.tick_accumulator += dt;
        while self.tick_accumulator >= self.tick_interval {
            self.tick_accumulator -= self.tick_interval;
            self.world.tick();
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, state: ElementState) -> bool {
        if self.camera_controller.process_keyboard(key, state) {
            return true;
        }
        if state == ElementState::Pressed {
            match key {
                KeyCode::F4 => {
                    self.renderer_config.disable_visibilty ^= true;
                    true
                }
                KeyCode::F3 => {
                    self.renderer_config.wireframe_mode ^= true;
                    true
                }
                KeyCode::F2 => {
                    self.renderer_config.render_aabbs ^= true;
                    true
                }
                _ => false,
            }
        } else {
            false
        }
    }

    pub fn handle_mouse_scroll(&mut self, delta: &MouseScrollDelta) {
        self.camera_controller.handle_mouse_scroll(delta);
    }

    pub fn handle_mouse(&mut self, dx: f64, dy: f64) {
        self.camera_controller.handle_mouse(dx, dy);
    }

    pub fn draw_frame(&mut self, cmd_rx: &Receiver<WorldUpdate>) {
        while let Ok(spos) = cmd_rx.try_recv() {
            self.update_world(spos);
        }
        let device = self.context.device();
        let frame = self.sync.next_frame();

        self.sync.wait_for_fence(device, frame);
        self.sync.process_deletion_queue(&self.context, frame);
        self.world
            .update_visibility(&self.context, frame, self.camera.position);

        let device = self.context.device();

        let image_index = match self
            .render_targets
            .swapchain
            .acquire_next_image(&self.sync, frame)
        {
            Ok(idx) => idx,
            Err(true) => {
                self.should_recreate = true;
                return;
            }
            Err(false) => panic!("Failed to acquire swapchain image"),
        };

        let cmd = self.command_buffers[frame];
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd, &begin_info).unwrap();
        }

        self.timestamp_pools
            .as_mut()
            .map(|arr| arr[frame].reset(device, cmd, 0, timings::TIMESTAMP_COUNT as u32));

        let mut frame_ctx = FrameCtx {
            ctx: &self.context,
            cmd,
            image_index,
            view_proj: self.projection.calc_proj() * self.camera.calc_view(),
            camera_pos: self.camera.position,
            frame_index: frame,
            config: self.renderer_config,
            timestamps: self.timestamp_pools.as_ref().map(|arr| &arr[frame]),
            frame_sync: &mut self.sync,
            render_targets: &self.render_targets,
        };
        frame_ctx.upload_to(
            &[Uniform {
                view_proj: frame_ctx.view_proj,
            }],
            &self.uniforms[frame_ctx.frame_index],
        );
        frame_ctx.begin_timestamp(timings::START_FRAME);

        self.world.render(&mut frame_ctx);
        frame_ctx.begin_timestamp(timings::START_UI_PASS);
        let dimensions = [
            self.render_targets.swapchain.extent.width,
            self.render_targets.swapchain.extent.height,
        ];

        if let Err(e) = self.egui.paint(
            &self.context,
            cmd,
            dimensions,
            image_index,
            frame_ctx.frame_index,
        ) {
            log::warn!("Failed to render egui: {}", e);
        }

        frame_ctx.begin_timestamp(timings::END_UI_PASS);

        frame_ctx.begin_timestamp(timings::END_FRAME);

        let device = self.context.device();

        unsafe {
            self.context.device().end_command_buffer(cmd).unwrap();
        }

        let wait_semaphores = [self.sync.image_available[frame]];
        let signal_semaphores = [self.sync.render_finished[image_index as usize]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd))
            .signal_semaphores(&signal_semaphores);

        unsafe {
            device
                .queue_submit(
                    self.context.graphics_queue(),
                    &[submit_info],
                    self.sync.in_flight[frame],
                )
                .unwrap();
        }

        match self.render_targets.swapchain.present(
            self.context.present_queue(),
            &self.sync,
            image_index,
        ) {
            Ok(true) => {}
            Ok(false) => self.should_recreate = true,
            Err(e) => panic!("Present failed: {:?}", e),
        }
    }

    /// Mark swapchain as invalid, to be recreated later.
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.projection.resize(size.width, size.height);
        if size.width > 0 && size.height > 0 {
            self.should_recreate = true;
            self.width = size.width;
            self.height = size.height;
        }
    }

    /// Actually recreate swapchain if marked.
    pub fn maybe_recreate(&mut self) {
        if self.should_recreate {
            unsafe {
                self.context
                    .device()
                    .queue_wait_idle(self.context.present_queue())
                    .unwrap();
                self.context
                    .device()
                    .queue_wait_idle(self.context.graphics_queue())
                    .unwrap();
            }
            self.render_targets
                .recreate(&self.context, self.width, self.height);

            // Let the world renderer handle its own swapchain recreation
            self.world
                .recreate_swapchain(&self.context, &self.render_targets);

            // Resize egui
            self.egui
                .resize(&self.context, &self.render_targets.swapchain);

            self.should_recreate = false;
        }
    }

    pub fn destroy(&mut self) {
        let device = self.context.device();

        unsafe {
            device.device_wait_idle().unwrap();

            self.timestamp_pools.as_ref().inspect(|pools| {
                pools.iter().for_each(|pool| {
                    pool.destroy(device);
                });
            });

            for uniform in &mut self.uniforms {
                uniform.destroy(&self.context);
            }

            device.destroy_command_pool(self.command_pool, None);
        }
        self.texture_manager.destroy(&self.context);

        self.world.destroy(&self.context);
        self.entity_renderer.destroy(&self.context);

        self.egui.destroy(&self.context);

        self.render_targets.destroy(&self.context);
        self.sync.destroy(&self.context);
    }

    /// Handle window events for egui.
    pub fn handle_egui_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.egui.on_window_event(window, event);
        response.consumed
    }
}

pub fn create_command_pool(ctx: &VkContext) -> vk::CommandPool {
    let device = ctx.device();
    let family_index = ctx.queue_families().graphics_index;

    let info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    unsafe { device.create_command_pool(&info, None).unwrap() }
}

pub fn allocate_command_buffers(
    ctx: &VkContext,
    pool: vk::CommandPool,
) -> [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT] {
    let device = ctx.device();

    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

    let mut buffers = [vk::CommandBuffer::null(); MAX_FRAMES_IN_FLIGHT];

    unsafe {
        (device.fp_v1_0().allocate_command_buffers)(
            device.handle(),
            &alloc_info,
            buffers.as_mut_ptr(),
        )
        .result()
        .unwrap()
    };

    buffers
}
