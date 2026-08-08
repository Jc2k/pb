use std::path::PathBuf;
use std::process::Command;

#[test]
fn tls_dependency_graph_excludes_external_crypto_toolchains() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(env!("CARGO"))
        .args([
            "tree",
            "--edges",
            "normal,build",
            "--prefix",
            "none",
            "--format",
            "{p}",
        ])
        .current_dir(root)
        .output()
        .expect("cargo tree should run");
    assert!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dependency_tree = String::from_utf8(output.stdout).expect("cargo tree should be UTF-8");

    for package in ["aws-lc-rs", "aws-lc-sys", "ring"] {
        assert!(
            !dependency_tree
                .lines()
                .any(|line| line.split_whitespace().next() == Some(package)),
            "TLS dependency graph unexpectedly contains {package}"
        );
    }
}
