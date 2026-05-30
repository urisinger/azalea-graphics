use std::rc::Rc;

use crate::context::CommandContext;

pub trait ResultConsumer<S, R> {
    fn on_command_complete(&self, context: Rc<CommandContext<S, R>>, success: bool, result: i32);
}

pub struct DefaultResultConsumer;
impl<S, R> ResultConsumer<S, R> for DefaultResultConsumer {
    fn on_command_complete(
        &self,
        _context: Rc<CommandContext<S, R>>,
        _success: bool,
        _result: i32,
    ) {
    }
}
