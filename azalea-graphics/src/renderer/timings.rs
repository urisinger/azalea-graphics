pub const TIMESTAMP_COUNT: usize = 12;

// Frame
pub const START_FRAME: usize = 0;
pub const END_FRAME: usize = 1;

// Upload dirty textures
pub const START_UPLOAD_DIRTY: usize = 2;
pub const END_UPLOAD_DIRTY: usize = 3;

// Terrain render pass
pub const START_TERRAIN_PASS: usize = 4;
pub const END_TERRAIN_PASS: usize = 5;

// HiZ compute
pub const START_HIZ_COMPUTE: usize = 6;
pub const END_HIZ_COMPUTE: usize = 7;

// Visibility compute
pub const START_VISIBILITY_COMPUTE: usize = 8;
pub const END_VISIBILITY_COMPUTE: usize = 9;

// UI pass (egui)
pub const START_UI_PASS: usize = 10;
pub const END_UI_PASS: usize = 11;

#[derive(Debug, Clone, Copy)]
pub struct Timings {
    ticks: [u64; TIMESTAMP_COUNT],
    timestamp_period: f32,
}

impl Timings {
    pub fn from_ticks(ticks: [u64; TIMESTAMP_COUNT], timestamp_period: f32) -> Self {
        Self { ticks, timestamp_period }
    }

    pub fn delta_ms(&self, start: usize, end: usize) -> f32 {
        let diff_ticks = self.ticks[end].saturating_sub(self.ticks[start]);
        (diff_ticks as f64 * self.timestamp_period as f64 / 1_000_000.0) as f32
    }

    pub fn frame_time(&self) -> f32 {
        self.delta_ms(START_FRAME, END_FRAME)
    }

    pub fn upload_dirty_time(&self) -> f32 {
        self.delta_ms(START_UPLOAD_DIRTY, END_UPLOAD_DIRTY)
    }

    pub fn terrain_pass_time(&self) -> f32 {
        self.delta_ms(START_TERRAIN_PASS, END_TERRAIN_PASS)
    }

    pub fn hiz_compute_time(&self) -> f32 {
        self.delta_ms(START_HIZ_COMPUTE, END_HIZ_COMPUTE)
    }

    pub fn visibility_compute_time(&self) -> f32 {
        self.delta_ms(START_VISIBILITY_COMPUTE, END_VISIBILITY_COMPUTE)
    }

    pub fn ui_time(&self) -> f32 {
        self.delta_ms(START_UI_PASS, END_UI_PASS)
    }
}
