use std::{io::Cursor, sync::Arc, time::Duration};

use ash::{
    util::read_spv,
    vk::{self, ShaderModuleCreateInfo},
};
use crossbeam::channel::Receiver;
use raw_window_handle::{DisplayHandle, WindowHandle};
use vulkan::{
    context::VkContext,
    frame_sync::{FrameSync, MAX_FRAMES_IN_FLIGHT},
    swapchain::Swapchain,
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
use crate::{app::WorldUpdate, renderer::world_renderer::WorldRendererConfig};

mod camera;
pub(crate) mod chunk;
mod mesh;
mod ui;
pub(crate) mod vulkan;
pub(crate) mod world_renderer;

mod assets;

pub struct Renderer {
    pub context: VkContext,
    pub swapchain: Swapchain,
    should_recreate: bool,
    width: u32,
    height: u32,

    renderer_config: WorldRendererConfig,
    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],

    sync: FrameSync,

    world: WorldRenderer,

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
    ) -> anyhow::Result<Self> {
        let context = VkContext::new(window_handle, display_handle);
        let swapchain = Swapchain::new(&context, size.width, size.height);

        let assets = assets::load_assets(&context, "assets/minecraft");

        let spirv = read_spv(&mut Cursor::new(include_bytes!(env!("SHADERS")))).unwrap();
        let module = unsafe {
            context
                .device()
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spirv), None)
                .unwrap()
        };

        let world = WorldRenderer::new(
            Arc::new(assets),
            &context,
            module,
            &swapchain,
            WorldRendererFeatures {
                fill_mode_non_solid: context.features().fill_mode_non_solid,
            },
        );

        let command_pool = create_command_pool(&context);
        let command_buffers = allocate_command_buffers(&context, command_pool);

        let sync = FrameSync::new(context.device(), swapchain.images.len());

        let camera = Camera::new(glam::vec3(0.0, 250.0, 2.0), 0.0, 90.0);
        let projection = Projection::new(size.width, size.height, 90.0, 0.1, 1000.0);
        let camera_controller = CameraController::new(4.0, 1.0);

        let egui = EguiVulkan::new(event_loop, &context, module, &swapchain, None)?;

        Ok(Self {
            context,
            swapchain,
            should_recreate: false,
            width: size.width,
            height: size.height,
            renderer_config: Default::default(),

            command_pool,
            command_buffers,

            sync,
            world,
            camera,
            projection,
            camera_controller,

            egui,

            tick_accumulator: Duration::ZERO,
            tick_interval: Duration::from_millis(50),
        })
    }

    /// Run the built-in debug UI.
    pub fn run_debug_ui(&mut self, window: &Window, frame_time_ms: f64) {
        let wireframe_available = self.context.features().fill_mode_non_solid;

        self.egui.run(window, |ctx| {
            egui::Window::new("Debug Info").show(ctx, |ui| {
                ui.label(format!("Frame time: {:.2}ms", frame_time_ms));
                ui.label("Azalea Graphics Renderer");

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
                )
            });
        });
    }

    pub fn update_world(&mut self, update: WorldUpdate) {
        self.world.update(&self.context, update);
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
        self.world
            .update_visibility(&self.context, frame, self.camera.position);

        let device = self.context.device();

        let image_index = match self.swapchain.acquire_next_image(&self.sync, frame) {
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

        self.world.render(
            &self.context,
            cmd,
            image_index,
            self.swapchain.extent,
            self.projection.calc_proj() * self.camera.calc_view(),
            self.camera.position,
            frame,
            self.renderer_config,
        );

        if let Err(e) = self.render_egui(cmd, image_index, frame) {
            log::warn!("Failed to render egui: {}", e);
        }
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

        match self
            .swapchain
            .present(self.context.present_queue(), &self.sync, image_index)
        {
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
            self.swapchain
                .recreate(&self.context, self.width, self.height);

            // Let the world renderer handle its own swapchain recreation
            self.world
                .recreate_swapchain(&self.context, &self.swapchain);

            // Resize egui
            self.egui.resize(&self.context, &self.swapchain);

            self.should_recreate = false;
        }
    }

    pub fn destroy(&mut self) {
        let device = self.context.device();

        unsafe {
            device.device_wait_idle().unwrap();

            self.world.destroy(&self.context);

            device.destroy_command_pool(self.command_pool, None);
        }

        self.egui.destroy(&self.context);

        self.swapchain.destroy(device);
        self.sync.destroy(device);
    }

    /// Handle window events for egui.
    pub fn handle_egui_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.egui.on_window_event(window, event);
        response.consumed
    }

    /// Render egui to the given command buffer.
    pub fn render_egui(
        &mut self,
        cmd: vk::CommandBuffer,
        image_index: u32,
        frame_index: usize,
    ) -> anyhow::Result<()> {
        let dimensions = [self.swapchain.extent.width, self.swapchain.extent.height];
        self.egui
            .paint(&self.context, cmd, dimensions, image_index, frame_index)
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
