use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectEntry {
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
    for project in &mut projects {
        if project.repository_root.is_none() {
            project.repository_root = repository_root_for(Path::new(&project.path));
        }
    }
    sort_projects(&mut projects);
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
    let existing_notify = load_projects_from(registry_path)?
        .into_iter()
        .find(|project| project.name == name || project.path == path_string)
        .map(|project| project.notify_on_finish)
        .unwrap_or(false);

    let entry = ProjectEntry {
        name: name.clone(),
        path: path_string,
        repository_root: repository_root_for(&path),
        notify_on_finish: existing_notify,
    };

    let mut projects = load_projects_from(registry_path)?;
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
    std::fs::write(&path, content).with_context(|| format!("failed to write {}", path.display()))
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

fn sort_projects(projects: &mut [ProjectEntry]) {
    projects.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.path.cmp(&b.path)));
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
    fn old_project_records_deserialize_without_repository_root() {
        let entry: ProjectEntry = toml::from_str(
            r#"
name = "legacy"
path = "/tmp/legacy"
notify_on_finish = true
"#,
        )
        .unwrap();
        assert_eq!(entry.repository_root, None);
        assert!(entry.notify_on_finish);
    }
}
