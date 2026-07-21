use std::fs;
use std::path::{Path, PathBuf};

const RETIRED_VARIABLES: &[&str] = &[
    "PB_STATE_DIR",
    "PB_CACHE_DIR",
    "PB_LLAMA_SESSION_CACHE",
    "PB_LLAMA_SESSION_CACHE_MAX_BYTES",
    "PB_FLASHMOE_SESSION_CACHE",
    "PB_FLASHMOE_SESSION_CACHE_MAX_BYTES",
    "PB_FLASHMOE_MEMORY_SESSIONS",
    "PB_FLASHMOE_RESIDENT_MODELS",
    "PB_FLASHMOE_IDLE_SECONDS",
];

#[test]
fn production_environment_access_stays_inside_the_audited_boundary() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();
    for path in rust_sources(&root.join("src")) {
        let relative = path.strip_prefix(&root).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        for (index, line) in text.lines().enumerate() {
            let accesses_parent = [
                "std::env::var(",
                "std::env::var_os(",
                "std::env::set_var(",
                "std::env::remove_var(",
                "option_env!(",
            ]
            .iter()
            .any(|pattern| line.contains(pattern));
            if accesses_parent && !allowed_access(relative, line) {
                violations.push(format!(
                    "{}:{}: {}",
                    relative.display(),
                    index + 1,
                    line.trim()
                ));
            }
        }
    }

    let build = fs::read_to_string(root.join("build.rs")).unwrap();
    for (index, line) in build.lines().enumerate() {
        if (line.contains("env::var(") || line.contains("option_env!("))
            && !line.contains("CARGO_MANIFEST_DIR")
            && !line.contains("PB_GITHUB_CLIENT_ID")
        {
            violations.push(format!("build.rs:{}: {}", index + 1, line.trim()));
        }
    }

    assert!(
        violations.is_empty(),
        "parent environment access must be config/CLI owned or added to the audited boundary:\n{}",
        violations.join("\n")
    );
}

#[test]
fn retired_pb_environment_variables_do_not_return() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = rust_sources(&root.join("src"));
    for directory in ["docs", ".github", "scripts"] {
        files.extend(all_files(&root.join(directory)));
    }
    files.push(root.join("README.md"));
    files.push(root.join("build.rs"));

    let mut violations = Vec::new();
    for path in files {
        let text = fs::read_to_string(&path).unwrap();
        for retired in RETIRED_VARIABLES {
            if text.contains(retired) {
                violations.push(format!(
                    "{} contains retired environment variable {retired}",
                    path.strip_prefix(&root).unwrap().display()
                ));
            }
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

fn allowed_access(path: &Path, line: &str) -> bool {
    match path.to_string_lossy().as_ref() {
        "src/host_environment.rs" => !line.contains("PB_"),
        "src/energy/macos.rs" => line.contains("PB_TEST_SYSTEM_ENERGY_LOCK_PATH"),
        "src/lib.rs" => line.contains("option_env!(\"PB_GITHUB_CLIENT_ID\")"),
        _ => false,
    }
}

fn rust_sources(root: &Path) -> Vec<PathBuf> {
    all_files(root)
        .into_iter()
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect()
}

fn all_files(root: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .collect()
}
