use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static PROJECT_ID_FALLBACK_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectEntry {
    #[serde(default)]
    pub id: String,
    pub name: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository_root: Option<String>,
    #[serde(default)]
    pub notify_on_finish: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ProjectRegistry {
    #[serde(default)]
    projects: Vec<ProjectEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddProjectRequest {
    pub name: Option<String>,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveProjectRequest {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateProjectNotificationsRequest {
    pub notify_on_finish: bool,
}

pub fn registry_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir().context("cannot determine config directory")?;
    Ok(config_dir.join("pb").join("projects.toml"))
}

pub fn default_project_name(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .map(|name| name.to_string())
        .with_context(|| format!("cannot infer project name from {}", path.display()))
}

pub fn canonical_project_path(path: impl AsRef<Path>) -> Result<PathBuf> {
    path.as_ref()
        .canonicalize()
        .with_context(|| format!("failed to resolve project path {}", path.as_ref().display()))
}

pub fn load_projects() -> Result<Vec<ProjectEntry>> {
    let path = registry_path()?;
    load_projects_from(&path)
}

fn load_projects_from(path: &Path) -> Result<Vec<ProjectEntry>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let registry: ProjectRegistry =
        toml::from_str(&content).with_context(|| format!("failed to parse {}", path.display()))?;
    let mut projects = registry.projects;
    let mut assigned_ids = false;
    for project in &mut projects {
        if project.id.trim().is_empty() {
            project.id = new_project_id();
            assigned_ids = true;
        }
        if project.repository_root.is_none() {
            project.repository_root = repository_root_for(Path::new(&project.path));
        }
    }
    sort_projects(&mut projects);
    let mut project_ids = HashSet::new();
    let mut project_names = HashSet::new();
    let mut project_paths = HashSet::new();
    for project in &projects {
        validate_project_id(&project.id)?;
        validate_project_name(&project.name)?;
        if project.path.trim().is_empty() || !Path::new(&project.path).is_absolute() {
            bail!("project '{}' has an invalid path", project.name);
        }
        if !project_ids.insert(project.id.as_str()) {
            bail!("project registry repeats id '{}'", project.id);
        }
        if !project_names.insert(project.name.as_str()) {
            bail!("project registry repeats name '{}'", project.name);
        }
        if !project_paths.insert(project.path.as_str()) {
            bail!("project registry repeats path '{}'", project.path);
        }
    }
    if assigned_ids {
        save_projects_to(path, &projects)?;
    }
    Ok(projects)
}

pub fn add_project(request: AddProjectRequest) -> Result<ProjectEntry> {
    add_project_at(request, &registry_path()?)
}

fn add_project_at(request: AddProjectRequest, registry_path: &Path) -> Result<ProjectEntry> {
    let path = canonical_project_path(&request.path)?;
    let name = match request.name {
        Some(name) if !name.trim().is_empty() => name.trim().to_string(),
        _ => default_project_name(&path)?,
    };
    validate_project_name(&name)?;

    let path_string = path.to_string_lossy().into_owned();
    let mut projects = load_projects_from(registry_path)?;
    let same_name = projects.iter().find(|project| project.name == name);
    let same_path = projects.iter().find(|project| project.path == path_string);
    if let (Some(named), Some(located)) = (same_name, same_path)
        && named.id != located.id
    {
        bail!(
            "project name '{}' and path '{}' identify different registered projects",
            name,
            path.display()
        );
    }
    let existing = same_name.or(same_path);
    let id = existing
        .map(|project| project.id.clone())
        .unwrap_or_else(new_project_id);
    let existing_notify = existing.is_some_and(|project| project.notify_on_finish);

    let entry = ProjectEntry {
        id,
        name: name.clone(),
        path: path_string,
        repository_root: repository_root_for(&path),
        notify_on_finish: existing_notify,
    };

    projects.retain(|project| project.name != name && project.path != entry.path);
    projects.push(entry.clone());
    sort_projects(&mut projects);
    save_projects_to(registry_path, &projects)?;
    Ok(entry)
}

pub fn remove_project(name: &str) -> Result<ProjectEntry> {
    validate_project_name(name)?;
    let mut projects = load_projects()?;
    let Some(index) = projects.iter().position(|project| project.name == name) else {
        bail!("project not found: {name}");
    };
    let removed = projects.remove(index);
    save_projects(&projects)?;
    Ok(removed)
}

pub fn set_project_notifications(name: &str, notify_on_finish: bool) -> Result<ProjectEntry> {
    set_project_notifications_at(&registry_path()?, name, notify_on_finish)
}

pub fn set_project_notifications_by_id(id: &str, notify_on_finish: bool) -> Result<ProjectEntry> {
    set_project_notifications_by_id_at(&registry_path()?, id, notify_on_finish)
}

fn set_project_notifications_by_id_at(
    registry_path: &Path,
    id: &str,
    notify_on_finish: bool,
) -> Result<ProjectEntry> {
    let mut projects = load_projects_from(registry_path)?;
    let Some(project) = projects.iter_mut().find(|project| project.id == id) else {
        bail!("project not found: {id}");
    };
    project.notify_on_finish = notify_on_finish;
    let updated = project.clone();
    save_projects_to(registry_path, &projects)?;
    Ok(updated)
}

fn set_project_notifications_at(
    registry_path: &Path,
    name: &str,
    notify_on_finish: bool,
) -> Result<ProjectEntry> {
    validate_project_name(name)?;
    let mut projects = load_projects_from(registry_path)?;
    let Some(project) = projects.iter_mut().find(|project| project.name == name) else {
        bail!("project not found: {name}");
    };
    project.notify_on_finish = notify_on_finish;
    let updated = project.clone();
    save_projects_to(registry_path, &projects)?;
    Ok(updated)
}

fn save_projects(projects: &[ProjectEntry]) -> Result<()> {
    let path = registry_path()?;
    save_projects_to(&path, projects)
}

fn save_projects_to(path: &Path, projects: &[ProjectEntry]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let content = toml::to_string_pretty(&ProjectRegistry {
        projects: projects.to_vec(),
    })?;
    let parent = path
        .parent()
        .with_context(|| format!("project registry path has no parent: {}", path.display()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to create a temporary file in {}", parent.display()))?;
    temporary.write_all(content.as_bytes()).with_context(|| {
        format!(
            "failed to write temporary project registry for {}",
            path.display()
        )
    })?;
    temporary.as_file().sync_all().with_context(|| {
        format!(
            "failed to sync temporary project registry for {}",
            path.display()
        )
    })?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

fn validate_project_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        bail!("project name cannot be empty");
    }
    if name.contains('\n') || name.contains('\r') {
        bail!("project name cannot contain newlines");
    }
    Ok(())
}

fn validate_project_id(id: &str) -> Result<()> {
    if !id.starts_with("project-")
        || id.len() == "project-".len()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        bail!("invalid project id '{id}'");
    }
    Ok(())
}

fn sort_projects(projects: &mut [ProjectEntry]) {
    projects.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.path.cmp(&b.path)));
}

fn new_project_id() -> String {
    let mut random = [0_u8; 16];
    if getrandom::getrandom(&mut random).is_ok() {
        use std::fmt::Write as _;
        let mut suffix = String::with_capacity(random.len() * 2);
        for byte in random {
            write!(&mut suffix, "{byte:02x}").expect("writing to a String cannot fail");
        }
        return format!("project-{suffix}");
    }
    let sequence = PROJECT_ID_FALLBACK_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!(
        "project-{}-{}-{sequence}",
        std::process::id(),
        crate::session_store::now_millis()
    )
}

fn repository_root_for(path: &Path) -> Option<String> {
    crate::agent_core::find_git_root(path).map(|root| {
        root.canonicalize()
            .unwrap_or(root)
            .to_string_lossy()
            .into_owned()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_name_uses_folder_name() {
        assert_eq!(
            default_project_name(Path::new("/tmp/example-project")).unwrap(),
            "example-project"
        );
    }

    #[test]
    fn notification_toggle_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let registry = dir.path().join("projects.toml");
        let project_dir = dir.path().join("example");
        std::fs::create_dir(&project_dir).unwrap();

        add_project_at(
            AddProjectRequest {
                name: Some("example".to_string()),
                path: project_dir.to_string_lossy().to_string(),
            },
            &registry,
        )
        .unwrap();
        let updated = set_project_notifications_at(&registry, "example", true).unwrap();
        assert!(updated.notify_on_finish);
        assert!(load_projects_from(&registry).unwrap()[0].notify_on_finish);

        let updated = set_project_notifications_by_id_at(&registry, &updated.id, false).unwrap();
        assert!(!updated.notify_on_finish);
    }

    #[test]
    fn project_identity_survives_registry_upgrade_and_repository_move() {
        let dir = tempfile::tempdir().unwrap();
        let registry = dir.path().join("projects.toml");
        let first_path = dir.path().join("first");
        let second_path = dir.path().join("second");
        std::fs::create_dir_all(&first_path).unwrap();
        std::fs::create_dir_all(&second_path).unwrap();
        std::fs::write(
            &registry,
            format!(
                "[[projects]]\nname = \"example\"\npath = \"{}\"\nnotify_on_finish = false\n",
                first_path.display()
            ),
        )
        .unwrap();

        let upgraded = load_projects_from(&registry).unwrap();
        assert!(upgraded[0].id.starts_with("project-"));
        assert_eq!(load_projects_from(&registry).unwrap()[0].id, upgraded[0].id);

        let moved = add_project_at(
            AddProjectRequest {
                name: Some("example".to_string()),
                path: second_path.to_string_lossy().into_owned(),
            },
            &registry,
        )
        .unwrap();
        assert_eq!(moved.id, upgraded[0].id);
        assert_eq!(
            moved.path,
            second_path.canonicalize().unwrap().to_string_lossy()
        );
    }

    #[test]
    fn project_update_rejects_ambiguous_existing_identities() {
        let dir = tempfile::tempdir().unwrap();
        let registry = dir.path().join("projects.toml");
        let first_path = dir.path().join("first");
        let second_path = dir.path().join("second");
        std::fs::create_dir_all(&first_path).unwrap();
        std::fs::create_dir_all(&second_path).unwrap();
        save_projects_to(
            &registry,
            &[
                ProjectEntry {
                    id: "project-first".to_string(),
                    name: "first".to_string(),
                    path: first_path
                        .canonicalize()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    repository_root: None,
                    notify_on_finish: false,
                },
                ProjectEntry {
                    id: "project-second".to_string(),
                    name: "second".to_string(),
                    path: second_path
                        .canonicalize()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    repository_root: None,
                    notify_on_finish: false,
                },
            ],
        )
        .unwrap();

        let error = add_project_at(
            AddProjectRequest {
                name: Some("first".to_string()),
                path: second_path.to_string_lossy().into_owned(),
            },
            &registry,
        )
        .unwrap_err();

        assert!(error.to_string().contains("different registered projects"));
        assert_eq!(load_projects_from(&registry).unwrap().len(), 2);
    }

    #[test]
    fn duplicate_registry_identity_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let registry = dir.path().join("projects.toml");
        std::fs::write(
            &registry,
            r#"
[[projects]]
id = "project-shared"
name = "first"
path = "/tmp/first"

[[projects]]
id = "project-shared"
name = "second"
path = "/tmp/second"
"#,
        )
        .unwrap();

        let error = load_projects_from(&registry).unwrap_err();
        assert!(error.to_string().contains("repeats id 'project-shared'"));
    }

    #[test]
    fn nested_project_keeps_focus_and_records_repository_root() {
        let repo = tempfile::tempdir().unwrap();
        assert!(
            std::process::Command::new("git")
                .args(["init"])
                .current_dir(repo.path())
                .status()
                .unwrap()
                .success()
        );
        let nested = repo.path().join("services").join("payments");
        std::fs::create_dir_all(&nested).unwrap();
        let entry = ProjectEntry {
            id: "project-payments".to_string(),
            name: "payments".to_string(),
            path: nested
                .canonicalize()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            repository_root: repository_root_for(&nested),
            notify_on_finish: false,
        };

        assert!(entry.path.ends_with("services/payments"));
        assert_eq!(
            entry.repository_root.as_deref(),
            Some(
                repo.path()
                    .canonicalize()
                    .unwrap()
                    .to_string_lossy()
                    .as_ref()
            )
        );
    }

    #[test]
    fn project_records_deserialize_without_derived_registry_fields() {
        let entry: ProjectEntry = toml::from_str(
            r#"
name = "legacy"
path = "/tmp/legacy"
notify_on_finish = true
"#,
        )
        .unwrap();
        assert_eq!(entry.id, "");
        assert_eq!(entry.repository_root, None);
        assert!(entry.notify_on_finish);
    }
}
