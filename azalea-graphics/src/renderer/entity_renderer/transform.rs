use azalea_assets::processed::entity_model::{Model, ModelPart, Transform};
use glam::{Mat4, Vec3};

pub struct ModelTransforms {
    transforms: Vec<Transform>,
}

impl ModelTransforms {
    pub fn new(model: &Model) -> Self {
        Self {
            transforms: model.default_transforms.clone(),
        }
    }

    pub fn to_transforms(&self, model: &Model, transform: Mat4) -> Vec<Mat4> {
        let mut transforms = vec![Mat4::IDENTITY; self.transforms.len()];
        self.to_transforms_helper(&mut transforms, transform, &model.part);
        transforms
    }

    fn to_transforms_helper(&self, transforms: &mut Vec<Mat4>, parent: Mat4, part: &ModelPart) {
        let transform = parent * self.transforms[part.id].to_mat4();
        transforms[part.id] = transform;
        for child in part.children.values() {
            self.to_transforms_helper(transforms, transform, child);
        }
    }

    /// Get immutable transform for a specific part
    pub fn get(&self, part: &ModelPart) -> &Transform {
        &self.transforms[part.id]
    }

    /// Get mutable transform for a specific part
    pub fn get_mut(&mut self, part: &ModelPart) -> &mut Transform {
        &mut self.transforms[part.id]
    }

    /// Get a part accessor by name from root
    pub fn part<'a>(&'a self, model: &'a ModelPart, name: &str) -> Option<PartAccessor<'a>> {
        model.children.get(name).map(|part| PartAccessor {
            transforms: &self.transforms,
            part,
        })
    }

    /// Get a mutable part accessor by name from root
    pub fn part_mut<'a>(
        &'a mut self,
        model: &'a ModelPart,
        name: &str,
    ) -> Option<PartAccessorMut<'a>> {
        model.children.get(name).map(|part| PartAccessorMut {
            transforms: &mut self.transforms,
            part,
        })
    }

    pub fn reset(&mut self, model: &Model) {
        self.transforms.copy_from_slice(&model.default_transforms);
    }

    pub fn as_slice(&self) -> &[Transform] {
        &self.transforms
    }
}

/// Immutable accessor for navigating and reading part transforms
pub struct PartAccessor<'a> {
    transforms: &'a [Transform],
    part: &'a ModelPart,
}

impl<'a> PartAccessor<'a> {
    /// Navigate to a child part
    pub fn child(&self, name: &str) -> Option<PartAccessor<'a>> {
        self.part.children.get(name).map(|part| PartAccessor {
            transforms: self.transforms,
            part,
        })
    }

    /// Get this part's transform
    pub fn transform(&self) -> &Transform {
        &self.transforms[self.part.id]
    }
}

/// Mutable accessor for navigating and modifying part transforms
pub struct PartAccessorMut<'a> {
    transforms: &'a mut [Transform],
    part: &'a ModelPart,
}

impl<'a> PartAccessorMut<'a> {
    /// Navigate to a child part
    pub fn child_mut(&mut self, name: &str) -> Option<PartAccessorMut<'_>> {
        self.part.children.get(name).map(|part| PartAccessorMut {
            transforms: self.transforms,
            part,
        })
    }

    pub fn transform_mut(&mut self) -> &mut Transform {
        &mut self.transforms[self.part.id]
    }

    pub fn set_angles(&mut self, angles: Vec3) {
        self.transforms[self.part.id].rotation = angles;
    }

    pub fn set_origin(&mut self, origin: Vec3) {
        self.transforms[self.part.id].pivot = origin;
    }
}
