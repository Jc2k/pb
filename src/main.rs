use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    pb::run(pb::Cli::parse()).await
}
