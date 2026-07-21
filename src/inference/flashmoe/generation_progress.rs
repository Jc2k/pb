use std::cell::RefCell;
use std::rc::Rc;

pub(super) type GenerationProgress<'a> = Option<Rc<RefCell<&'a mut dyn FnMut(String)>>>;

pub(super) fn report_generation_progress<F>(progress: &GenerationProgress<'_>, message: F)
where
    F: FnOnce() -> String,
{
    if let Some(callback) = progress {
        (callback.borrow_mut())(message());
    }
}
