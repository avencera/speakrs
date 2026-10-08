//! Command-line tools for CUDA tuning

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

use speakrs::inference::cuda::{CudaMath, CudaTuneOptions, CudaTuneReport, tune_cuda};

const USAGE: &str = "Usage: speakrs cuda tune --models-dir PATH [OPTIONS]

Measure CUDA settings and save the selected settings.

Options:
  --models-dir PATH          Model directory (required)
  --models PATH              Same as --models-dir
  --output PATH              Output file
  --device N                 CUDA device number
  --dry-run                  Measure without writing a file
  --include-library          Also time Library (excludes load and handle costs)
  --segmentation-math MODE   fp32 or tf32
  --embedding-math MODE      fp32 or tf32
  --help                    Show this help";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1).peekable();
    if args.peek().is_some_and(|arg| arg == "--help") {
        args.next();
        if args.next().is_some() {
            return Err("Do not add arguments after --help".into());
        }
        println!("{USAGE}");
        return Ok(());
    }

    for command in ["cuda", "tune"] {
        let arg = args
            .next()
            .ok_or_else(|| format!("Expected {command}. Use --help for help"))?;
        if arg == "--help" {
            if args.next().is_some() {
                return Err("Do not add arguments after --help".into());
            }
            println!("{USAGE}");
            return Ok(());
        }
        if arg != command {
            return Err(format!("Expected {command}, got {}", arg.to_string_lossy()));
        }
    }

    let mut options = CudaTuneOptions::default();
    let mut model_dir = None;
    while let Some(arg) = args.next() {
        let flag = arg.to_str().ok_or("Option names must use valid UTF-8")?;
        match flag {
            "--models-dir" | "--models" => model_dir = Some(PathBuf::from(value(&mut args, flag)?)),
            "--output" => options.output_path = Some(PathBuf::from(value(&mut args, flag)?)),
            "--device" => {
                let input = value(&mut args, flag)?;
                options.device = input
                    .to_str()
                    .and_then(|text| text.parse::<usize>().ok())
                    .ok_or("--device must be a non-negative whole number")?;
            }
            "--dry-run" => options.dry_run = true,
            "--include-library" => options.include_library = true,
            "--segmentation-math" => {
                options.segmentation_math = math(value(&mut args, flag)?, flag)?
            }
            "--embedding-math" => options.embedding_math = math(value(&mut args, flag)?, flag)?,
            "--help" => {
                if args.next().is_some() {
                    return Err("Do not add arguments after --help".into());
                }
                println!("{USAGE}");
                return Ok(());
            }
            _ if flag.starts_with('-') => return Err(format!("Unknown option: {flag}")),
            _ => return Err(format!("Unexpected argument: {flag}")),
        }
    }

    options.model_dir = model_dir.ok_or("--models-dir PATH is required")?;
    let report = tune_cuda(&options).map_err(|error| error.to_string())?;
    print_report(&report);
    Ok(())
}

fn value(args: &mut impl Iterator<Item = OsString>, flag: &str) -> Result<OsString, String> {
    let value = args
        .next()
        .ok_or_else(|| format!("Missing value for {flag}"))?;
    if value.is_empty() || value.to_str().is_some_and(|text| text.starts_with('-')) {
        return Err(format!("Missing value for {flag}"));
    }
    Ok(value)
}

fn math(value: OsString, flag: &str) -> Result<CudaMath, String> {
    match value.to_str() {
        Some("fp32") => Ok(CudaMath::Fp32),
        Some("tf32") => Ok(CudaMath::Tf32),
        _ => Err(format!("{flag} must be fp32 or tf32")),
    }
}

fn print_report(report: &CudaTuneReport) {
    println!("Device: {}", report.device);
    println!("Output: {}", report.path.display());
    let boundary_width = report
        .rows
        .iter()
        .map(|row| row.boundary.len())
        .max()
        .unwrap_or(0)
        .max(8);
    let candidate_width = report
        .rows
        .iter()
        .flat_map(|row| &row.candidates)
        .map(|measurement| measurement.candidate.len())
        .max()
        .unwrap_or(0)
        .max(9);
    println!(
        "{:<boundary_width$} {:>5} {:<7} {:<candidate_width$} {:>12}  Selected",
        "Boundary", "Batch", "Math", "Candidate", "Median (ms)"
    );
    for row in &report.rows {
        let math = match row.math {
            CudaMath::Fp32 => "fp32",
            CudaMath::Tf32 => "tf32",
            _ => "unknown",
        };
        for measurement in &row.candidates {
            let selected = if measurement.candidate == row.selected {
                "*"
            } else {
                ""
            };
            println!(
                "{:<boundary_width$} {:>5} {:<7} {:<candidate_width$} {:>12.3}  {selected}",
                row.boundary, row.batch, math, measurement.candidate, measurement.median_ms
            );
        }
    }

    if report.written {
        println!("Wrote file {}", report.path.display());
    } else {
        println!("Dry run: no file was written");
    }
}
