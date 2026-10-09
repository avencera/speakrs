use clap::Parser;
use color_eyre::eyre::Result;
use tracing_subscriber::EnvFilter;
use xtask::cli::Cli;

fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    Cli::parse().run()
}
