use super::{Suggestions, SuggestionsBuilder};
use crate::context::CommandContext;

pub trait SuggestionProvider<S, R> {
    fn get_suggestions(
        &self,
        context: CommandContext<S, R>,
        builder: SuggestionsBuilder,
    ) -> Suggestions;
}
