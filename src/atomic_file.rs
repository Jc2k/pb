use anyhow::{Context, Result};
use std::io::Write;
use std::path::Path;

pub(crate) fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create {}", parent.display()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to create temporary file in {}", parent.display()))?;
    temporary
        .write_all(bytes)
        .with_context(|| format!("failed to stage {}", path.display()))?;
    temporary
        .as_file_mut()
        .sync_all()
        .with_context(|| format!("failed to sync staged {}", path.display()))?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to atomically replace {}", path.display()))?;
    if let Ok(directory) = std::fs::File::open(parent) {
        directory
            .sync_all()
            .with_context(|| format!("failed to sync {}", parent.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_replaces_existing_content() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        std::fs::write(&path, "old").unwrap();
        write(&path, b"new").unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), "new");
    }
}
