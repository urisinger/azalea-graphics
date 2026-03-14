use azalea::{
    ecs::{entity::Entity, world::World},
    entity::{EntityKindComponent, Position, metadata::Invisible},
    physics::collision::VoxelShape,
    registry::EntityKind,
};
use glam::Vec3;

#[derive(Debug, Clone)]
pub struct EntityRenderState {
    pub kind: EntityKind,
    pub pos: Vec3,
    pub age: f32,
    pub width: f32,
    pub height: f32,
    pub standing_eye_height: f32,
    pub squared_distance_to_camera: f64,
    pub invisible: bool,
    pub sneaking: bool,
    pub on_fire: bool,
    pub light: i32,
    pub outline_color: i32,
    pub position_offset: Option<Vec3>,
    //pub display_name: Option<Text>,
    pub name_label_pos: Option<Vec3>,
    pub leash_datas: Option<Vec<LeashData>>,
    pub shadow_radius: f32,
    pub shadow_pieces: Vec<ShadowPiece>,
}

#[derive(Debug, Clone)]
pub struct LeashData {
    pub offset: Vec3,
    pub start_pos: Vec3,
    pub end_pos: Vec3,
    pub leashed_entity_block_light: i32,
    pub leash_holder_block_light: i32,
    pub leashed_entity_sky_light: i32,
    pub leash_holder_sky_light: i32,
    pub slack: bool,
}

impl Default for LeashData {
    fn default() -> Self {
        Self {
            offset: Vec3::ZERO,
            start_pos: Vec3::ZERO,
            end_pos: Vec3::ZERO,
            leashed_entity_block_light: 0,
            leash_holder_block_light: 0,
            leashed_entity_sky_light: 15,
            leash_holder_sky_light: 15,
            slack: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShadowPiece {
    pub relative_x: f32,
    pub relative_y: f32,
    pub relative_z: f32,
    pub shape_below: VoxelShape,
    pub alpha: f32,
}

impl EntityRenderState {
    pub fn new(world: &mut World, entity: Entity) -> Self {
        let pos: &Position = world.get(entity).unwrap();
        let kind: &EntityKindComponent = world.get(entity).unwrap();
        let invisible: &Invisible = world.get(entity).unwrap();
        Self {
            kind: kind.0,
            pos: Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
            invisible: invisible.0,
            age: 0.0,
            width: 0.0,
            height: 0.0,
            standing_eye_height: 0.0,
            squared_distance_to_camera: 0.0,
            sneaking: false,
            on_fire: false,
            light: 0,
            outline_color: 0,
            position_offset: None,
            //display_name: None,
            name_label_pos: None,
            leash_datas: None,
            shadow_radius: 0.0,
            shadow_pieces: Vec::new(),
        }
    }
}

// Optional methods from Java
impl EntityRenderState {
    pub fn has_outline(&self) -> bool {
        self.outline_color != 0
    }
}
