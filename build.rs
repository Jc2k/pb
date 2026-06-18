fn main() {
    println!("cargo:rerun-if-changed=webui/dist");
    println!("cargo:rerun-if-env-changed=PB_GITHUB_CLIENT_ID");
}
