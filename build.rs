use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=webui/src/index.html");
    println!("cargo:rerun-if-changed=webui/src/app.css");
    println!("cargo:rerun-if-env-changed=PB_GITHUB_CLIENT_ID");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set"));
    let dist = manifest_dir.join("webui/dist");
    if dist.join("index.html").exists() {
        return;
    }

    fs::create_dir_all(&dist).expect("create webui/dist fallback directory");
    fs::write(
        dist.join("index.html"),
        r##"<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
    <meta name="apple-mobile-web-app-capable" content="yes" />
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
    <meta name="theme-color" content="#111120" />
    <link rel="manifest" href="/manifest.json" />
    <title>pb serve</title>
    <style>
      body { margin: 0; padding: env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left); font-family: system-ui, sans-serif; background: #111120; color: #f8f8ff; }
      main { max-width: 48rem; margin: 15vh auto; padding: 2rem; }
      code { background: #252540; padding: 0.15rem 0.35rem; border-radius: 0.25rem; }
    </style>
  </head>
  <body>
    <main>
      <h1>pb serve</h1>
      <p>The bundled web UI assets were not built in this checkout.</p>
      <p>Install Deno and run <code>deno task build:web</code> to generate the full React UI.</p>
    </main>
  </body>
</html>
"##,
    )
    .expect("write fallback web UI index");
    fs::write(
        dist.join("manifest.json"),
        r##"{"name":"pb serve","short_name":"pb","display":"standalone","start_url":"/","theme_color":"#111120","background_color":"#111120"}"##,
    )
    .expect("write fallback web manifest");
}
