use std::{
    collections::BTreeMap,
    io,
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
    },
};

use ruff_db::{
    file_revision::FileRevision,
    system::{
        DirectoryEntry, FileType, MemoryFileSystem, Metadata, System, SystemPath, SystemPathBuf,
        SystemVirtualPath, WhichError, WhichResult, walk_directory::WalkDirectoryBuilder,
    },
};
use ruff_notebook::{Notebook, NotebookError};

#[derive(Clone, Debug)]
struct OverlayFile {
    text: Arc<str>,
    revision: FileRevision,
}

#[derive(Clone, Debug)]
enum OverlayEntry {
    File(OverlayFile),
    Deleted,
}

/// A frozen project image with a request-private copy-on-write overlay. The base image is created
/// before inference. A fork shares only immutable bytes; generated files and revisions are never
/// visible to another request or to the warm project world.
#[derive(Clone, Debug)]
pub(crate) struct PythonSystem {
    base: MemoryFileSystem,
    overlays: Arc<RwLock<BTreeMap<SystemPathBuf, OverlayEntry>>>,
    next_revision: Arc<AtomicU64>,
}

impl PythonSystem {
    pub(crate) fn from_base(base: MemoryFileSystem) -> Self {
        Self {
            base,
            overlays: Arc::new(RwLock::new(BTreeMap::new())),
            next_revision: Arc::new(AtomicU64::new(1)),
        }
    }

    pub(crate) fn fork(&self) -> Self {
        Self {
            base: self.base.clone(),
            overlays: Arc::new(RwLock::new(BTreeMap::new())),
            next_revision: Arc::new(AtomicU64::new(1)),
        }
    }

    pub(crate) fn put(&self, path: &SystemPath, text: String) -> io::Result<()> {
        let path = self.absolute(path);
        if path.as_path() == self.current_directory() || !path.starts_with(self.current_directory())
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Python overlay path escapes the frozen project root",
            ));
        }
        let revision = self
            .next_revision
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        self.overlays.write().map_err(|_| poisoned())?.insert(
            path,
            OverlayEntry::File(OverlayFile {
                text: Arc::from(text),
                revision: FileRevision::from(revision),
            }),
        );
        Ok(())
    }

    pub(crate) fn delete(&self, path: &SystemPath) -> io::Result<()> {
        let path = self.absolute(path);
        if path.as_path() == self.current_directory() || !path.starts_with(self.current_directory())
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "Python overlay path escapes the frozen project root",
            ));
        }
        self.next_revision.fetch_add(1, Ordering::Relaxed);
        self.overlays
            .write()
            .map_err(|_| poisoned())?
            .insert(path, OverlayEntry::Deleted);
        Ok(())
    }

    pub(crate) fn reset(&self, path: &SystemPath) -> io::Result<()> {
        let path = self.absolute(path);
        self.next_revision.fetch_add(1, Ordering::Relaxed);
        self.overlays.write().map_err(|_| poisoned())?.remove(&path);
        Ok(())
    }

    fn absolute(&self, path: &SystemPath) -> SystemPathBuf {
        SystemPath::absolute(path, self.base.current_directory())
    }

    fn overlay_entry(&self, path: &SystemPath) -> io::Result<Option<OverlayEntry>> {
        let path = self.absolute(path);
        Ok(self
            .overlays
            .read()
            .map_err(|_| poisoned())?
            .get(&path)
            .cloned())
    }

    fn overlay_has_descendant(&self, path: &SystemPath) -> io::Result<bool> {
        let path = self.absolute(path);
        Ok(self
            .overlays
            .read()
            .map_err(|_| poisoned())?
            .iter()
            .any(|(candidate, entry)| {
                matches!(entry, OverlayEntry::File(_))
                    && candidate != &path
                    && candidate.starts_with(&path)
            }))
    }
}

impl System for PythonSystem {
    fn path_metadata(&self, path: &SystemPath) -> io::Result<Metadata> {
        match self.overlay_entry(path)? {
            Some(OverlayEntry::File(file)) => {
                return Ok(Metadata::new(file.revision, Some(0o444), FileType::File));
            }
            Some(OverlayEntry::Deleted) => return Err(not_found()),
            None => {}
        }
        if self.overlay_has_descendant(path)? {
            return Ok(Metadata::new(
                FileRevision::from(self.next_revision.load(Ordering::Relaxed)),
                Some(0o555),
                FileType::Directory,
            ));
        }
        self.base.metadata(path)
    }

    fn canonicalize_path(&self, path: &SystemPath) -> io::Result<SystemPathBuf> {
        let absolute = self.absolute(path);
        self.path_metadata(&absolute)?;
        Ok(absolute)
    }

    fn is_same_file(&self, first: &SystemPath, second: &SystemPath) -> io::Result<bool> {
        Ok(self.canonicalize_path(first)? == self.canonicalize_path(second)?)
    }

    fn read_to_string(&self, path: &SystemPath) -> io::Result<String> {
        match self.overlay_entry(path)? {
            Some(OverlayEntry::File(file)) => return Ok(file.text.to_string()),
            Some(OverlayEntry::Deleted) => return Err(not_found()),
            None => {}
        }
        self.base.read_to_string(path)
    }

    fn read_to_notebook(&self, path: &SystemPath) -> Result<Notebook, NotebookError> {
        Notebook::from_source_code(&self.read_to_string(path)?)
    }

    fn read_virtual_path_to_string(&self, _path: &SystemVirtualPath) -> io::Result<String> {
        Err(not_found())
    }

    fn read_virtual_path_to_notebook(
        &self,
        _path: &SystemVirtualPath,
    ) -> Result<Notebook, NotebookError> {
        Err(NotebookError::from(not_found()))
    }

    fn current_directory(&self) -> &SystemPath {
        self.base.current_directory()
    }

    fn user_config_directory(&self) -> Option<SystemPathBuf> {
        None
    }

    fn cache_dir(&self) -> Option<SystemPathBuf> {
        None
    }

    fn which(&self, _name: &str) -> WhichResult {
        Err(WhichError::CannotFindBinaryPath)
    }

    fn read_directory<'a>(
        &'a self,
        path: &SystemPath,
    ) -> io::Result<Box<dyn Iterator<Item = io::Result<DirectoryEntry>> + 'a>> {
        let parent = self.absolute(path);
        let mut entries = BTreeMap::new();
        if let Ok(base) = self.base.read_directory(&parent) {
            for entry in base {
                let entry = entry?;
                entries.insert(entry.path().to_path_buf(), entry.file_type());
            }
        }
        let overlays = self.overlays.read().map_err(|_| poisoned())?;
        for (candidate, overlay) in overlays.iter() {
            let Ok(relative) = candidate.strip_prefix(&parent) else {
                continue;
            };
            let mut components = relative.components();
            let Some(first) = components.next() else {
                continue;
            };
            let child = parent.join(first.as_str());
            let nested = components.next().is_some();
            if matches!(overlay, OverlayEntry::Deleted) && !nested {
                entries.remove(&child);
                continue;
            }
            if matches!(overlay, OverlayEntry::Deleted) {
                continue;
            }
            let file_type = if nested {
                FileType::Directory
            } else {
                FileType::File
            };
            entries
                .entry(child)
                .and_modify(|existing| {
                    if file_type == FileType::Directory {
                        *existing = FileType::Directory;
                    }
                })
                .or_insert(file_type);
        }
        let entries = entries
            .into_iter()
            .map(|(path, file_type)| Ok(DirectoryEntry::new(path, file_type)))
            .collect::<Vec<_>>();
        Ok(Box::new(entries.into_iter()))
    }

    fn walk_directory(&self, path: &SystemPath) -> WalkDirectoryBuilder {
        // Project discovery happens while building the frozen image. ty uses direct metadata and
        // directory reads for request overlays, so the immutable walker remains authoritative.
        self.base.walk_directory(path)
    }

    fn as_writable(&self) -> Option<&dyn ruff_db::system::WritableSystem> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn dyn_clone(&self) -> Box<dyn System> {
        Box::new(self.clone())
    }
}

fn poisoned() -> io::Error {
    io::Error::other("Python request overlay lock is poisoned")
}

fn not_found() -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, "virtual Python source not found")
}
