use std::sync::Arc;

use crate::{context::CommandContext, errors::CommandSyntaxError};

pub type RedirectModifier<S, R> =
    dyn Fn(&CommandContext<S, R>) -> Result<Vec<Arc<S>>, CommandSyntaxError> + Send + Sync;
