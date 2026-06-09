use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=webui/src/index.html");
    println!("cargo:rerun-if-changed=webui/src/app.js");
    println!("cargo:rerun-if-changed=webui/src/app.css");

    let src_dir = Path::new("webui/src");
    let dist_dir = Path::new("webui/dist");

    fs::create_dir_all(dist_dir).expect("failed to create webui/dist");
    for file in ["index.html", "app.js", "app.css"] {
        fs::copy(src_dir.join(file), dist_dir.join(file))
            .unwrap_or_else(|e| panic!("failed to stage {file}: {e}"));
    }
}
