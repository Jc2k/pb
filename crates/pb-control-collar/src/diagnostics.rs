use thiserror::Error;

pub type CollarResult<T> = Result<T, CollarError>;

#[derive(Debug, Error)]
pub enum CollarError {
    #[error("invalid collar manifest: {0}")]
    InvalidManifest(String),
    #[error("invalid tokenizer vocabulary: {0}")]
    InvalidVocabulary(String),
    #[error("tool protocol constraint failed: {0}")]
    Protocol(String),
    #[error("LLGuidance constraint failed: {0}")]
    Guidance(String),
    #[error("virtual mutation failed: {0}")]
    Mutation(String),
    #[error("source analysis failed: {0}")]
    Analysis(String),
    #[error("collar state is incomplete: {0}")]
    Incomplete(String),
}
