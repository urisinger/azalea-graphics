use std::{
    collections::HashMap,
    fmt::{self, Debug},
    rc::Rc,
};

use crate::{
    context::CommandContextBuilder, errors::CommandSyntaxError, string_reader::StringReader,
    tree::CommandNode,
};

pub struct ParseResults<'a, S, R> {
    pub context: CommandContextBuilder<'a, S, R>,
    pub reader: StringReader,
    pub exceptions: HashMap<Rc<CommandNode<S, R>>, CommandSyntaxError>,
}

impl<S, R> Debug for ParseResults<'_, S, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParseResults")
            .field("context", &self.context)
            // .field("reader", &self.reader)
            .field("exceptions", &self.exceptions)
            .finish()
    }
}
