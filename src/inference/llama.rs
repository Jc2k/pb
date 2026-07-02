use anyhow::{Context, Result};
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use std::sync::OnceLock;

pub fn init_backend() -> Result<LlamaBackend> {
    suppress_logs();
    let mut backend = LlamaBackend::init().context("failed to initialize llama backend")?;
    backend.void_logs();
    Ok(backend)
}

fn suppress_logs() {
    static LLAMA_LOGS_SUPPRESSED: OnceLock<()> = OnceLock::new();
    LLAMA_LOGS_SUPPRESSED.get_or_init(|| {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
    });
}

pub use llama_cpp_2::model::{AddBos, LlamaModel};
pub use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText, mtmd_default_marker,
};
pub use llama_cpp_2::sampling::LlamaSampler;
pub use llama_cpp_2::{
    context::params::LlamaContextParams, llama_backend::LlamaBackend, llama_batch::LlamaBatch,
    model::params::LlamaModelParams,
};
