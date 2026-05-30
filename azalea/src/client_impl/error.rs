use bevy_ecs::entity::Entity;
use thiserror::Error;

/// An error that occurs when we tried to access data from an entity that it
/// doesn't have.
///
/// This could happen because the data does not occur for this type of entity,
/// or because the entity is not currently loaded.
///
/// If this error happened when trying to access data on a client, then it may
/// be because the client isn't currently in a world.
///
/// As an alias for `Result<T, MissingComponentError>`, you may use
/// [`AzaleaResult`]. Using the `eyre` or `anyhow` crates may also simplify
/// error handling in your bots.
#[derive(Error, Debug)]
#[error("{entity_description} {entity} is missing a required component: '{component}'")]
pub struct MissingComponentError {
    /// Should be "Entity" or "Client"
    pub entity_description: &'static str,
    pub entity: Entity,
    pub component: &'static str,
}

/// An error that occurs when we tried to access data from an entity that it
/// doesn't have.
///
/// See [`MissingComponentError`] for more details.
pub type AzaleaResult<T> = Result<T, MissingComponentError>;
