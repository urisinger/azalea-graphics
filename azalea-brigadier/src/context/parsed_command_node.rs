use std::sync::Arc;

use parking_lot::RwLock;

use super::string_range::StringRange;
use crate::tree::CommandNode;

#[derive(Debug)]
pub struct ParsedCommandNode<S, R> {
    pub node: Arc<RwLock<CommandNode<S, R>>>,
    pub range: StringRange,
}

impl<S, R> Clone for ParsedCommandNode<S, R> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            range: self.range,
        }
    }
}
