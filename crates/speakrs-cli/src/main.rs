//! The `speakrs` command: diarize audio files to RTTM, and tune the CUDA backend

mod diarize;
#[cfg(feature = "_cuda")]
mod tune;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "speakrs",
    version,
    about = "Speaker diarization with RTTM output"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Diarize(diarize::DiarizeArgs),
    /// NVIDIA GPU tools
    #[cfg(feature = "_cuda")]
    Cuda {
        #[command(subcommand)]
        command: tune::CudaCommand,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Diarize(args) => diarize::run(args),
        #[cfg(feature = "_cuda")]
        Command::Cuda { command } => tune::run(command),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // the alternate form prints the cause chain on one line without eyre's
            // location report, which is noise for a command-line user
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
