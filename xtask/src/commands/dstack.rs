use std::io::{BufRead, Write};

use color_eyre::eyre::{Result, bail};
use xshell::{Shell, cmd};

use crate::cmd::project_root;

fn require_env(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| color_eyre::eyre::eyre!("{name} must be set"))
}

pub fn bench(
    name: &str,
    dataset: &str,
    impls: &[String],
    max_files: Option<u32>,
    max_minutes: Option<u32>,
    reuse: bool,
    detach: bool,
) -> Result<()> {
    require_env("AWS_ENDPOINT_URL")?;
    require_env("AWS_ACCESS_KEY_ID")?;

    let sh = Shell::new()?;
    sh.change_dir(project_root());

    sh.set_var("RUN_NAME", name);
    sh.set_var("DATASET", dataset);

    sh.set_var("IMPLS", if impls.is_empty() { String::new() } else { impls.join(",") });
    sh.set_var("MAX_FILES", max_files.map(|n| n.to_string()).unwrap_or_default());
    sh.set_var("MAX_MINUTES", max_minutes.map(|n| n.to_string()).unwrap_or_default());

    if reuse {
        // ensure fleet exists before submitting
        cmd!(sh, "dstack apply -f .dstack/fleet.yml -y").run()?;
    }

    let mut args = vec![
        "apply".to_string(),
        "-f".to_string(),
        ".dstack/benchmark.yml".to_string(),
        "-n".to_string(),
        name.to_string(),
        "-y".to_string(),
    ];

    if reuse {
        args.push("-R".to_string());
    }
    if detach {
        args.push("--detach".to_string());
    }

    let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    cmd!(sh, "dstack {args_ref...}").run()?;

    Ok(())
}

pub fn fleet() -> Result<()> {
    let sh = Shell::new()?;
    sh.change_dir(project_root());
    cmd!(sh, "dstack apply -f .dstack/fleet.yml -y").run()?;
    Ok(())
}

pub fn attach(name: &str) -> Result<()> {
    let sh = Shell::new()?;
    sh.change_dir(project_root());
    cmd!(sh, "dstack attach {name}").run()?;
    Ok(())
}

pub fn logs(name: &str) -> Result<()> {
    let sh = Shell::new()?;
    sh.change_dir(project_root());
    cmd!(sh, "dstack logs {name}").run()?;
    Ok(())
}

pub fn ps() -> Result<()> {
    let sh = Shell::new()?;
    sh.change_dir(project_root());
    cmd!(sh, "dstack ps").run()?;
    Ok(())
}

pub fn stop(name: &str) -> Result<()> {
    let sh = Shell::new()?;
    sh.change_dir(project_root());
    cmd!(sh, "dstack stop {name}").run()?;
    Ok(())
}

pub fn dev() -> Result<()> {
    let sh = Shell::new()?;
    sh.change_dir(project_root());
    cmd!(sh, "dstack apply -f .dstack/dev.yml -y").run()?;
    Ok(())
}

pub fn download(name: &str) -> Result<()> {
    let endpoint = require_env("AWS_ENDPOINT_URL")?;
    let sh = Shell::new()?;
    sh.change_dir(project_root());

    let local_dir = format!("_benchmarks/{name}");
    std::fs::create_dir_all(&local_dir)?;

    let src = format!("s3://speakrs/benchmarks/{name}/*");
    cmd!(
        sh,
        "s5cmd --endpoint-url {endpoint} cp {src} {local_dir}/"
    )
    .run()?;
    Ok(())
}

pub fn delete(path: &str) -> Result<()> {
    let endpoint = require_env("AWS_ENDPOINT_URL")?;
    let sh = Shell::new()?;
    sh.change_dir(project_root());

    let prefix = format!("s3://speakrs/{path}/*");

    // list objects first
    let listing = cmd!(sh, "s5cmd --endpoint-url {endpoint} ls {prefix}")
        .read()
        .unwrap_or_default();

    if listing.trim().is_empty() {
        println!("No objects found under s3://speakrs/{path}/");
        return Ok(());
    }

    let count = listing.lines().count();
    println!("{listing}");
    print!("\nDelete {count} objects? [y/N] ");
    std::io::stdout().flush()?;

    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;

    if answer.trim().eq_ignore_ascii_case("y") {
        cmd!(sh, "s5cmd --endpoint-url {endpoint} rm {prefix}").run()?;
        println!("Deleted");
    } else {
        bail!("Aborted");
    }

    Ok(())
}
