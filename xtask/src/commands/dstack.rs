use std::io::{BufRead, Write};

use color_eyre::eyre::{Result, bail};
use xshell::{Shell, cmd};

use crate::cmd::project_root;
use crate::datasets;

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

    sh.set_var(
        "IMPLS",
        if impls.is_empty() {
            String::new()
        } else {
            impls.join(",")
        },
    );
    sh.set_var(
        "MAX_FILES",
        max_files.map(|n| n.to_string()).unwrap_or_default(),
    );
    sh.set_var(
        "MAX_MINUTES",
        max_minutes.map(|n| n.to_string()).unwrap_or_default(),
    );

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

pub fn bench_parallel(
    name: &str,
    dataset: &[String],
    impls: &[String],
    max_files: Option<u32>,
    max_minutes: Option<u32>,
    reuse: bool,
) -> Result<()> {
    require_env("AWS_ENDPOINT_URL")?;
    require_env("AWS_ACCESS_KEY_ID")?;

    let dataset_ids: Vec<String> = if dataset.len() == 1 && dataset[0] == "all" {
        datasets::list_dataset_ids()
    } else {
        for id in dataset {
            if datasets::find_dataset(id).is_none() {
                bail!("unknown dataset: {id}. Use --dataset list on speakrs-bm to see available");
            }
        }
        dataset.to_vec()
    };

    let sh = Shell::new()?;
    sh.change_dir(project_root());

    if reuse {
        cmd!(sh, "dstack apply -f .dstack/fleet.yml -y").run()?;
    }

    let impls_str = if impls.is_empty() {
        String::new()
    } else {
        impls.join(",")
    };
    let max_files_str = max_files.map(|n| n.to_string()).unwrap_or_default();
    let max_minutes_str = max_minutes.map(|n| n.to_string()).unwrap_or_default();

    println!("Submitting {} tasks:", dataset_ids.len());

    let mut submitted = Vec::new();
    for ds_id in &dataset_ids {
        let task_name = format!("{name}-{ds_id}");

        sh.set_var("RUN_NAME", &task_name);
        sh.set_var("DATASET", ds_id);
        sh.set_var("IMPLS", &impls_str);
        sh.set_var("MAX_FILES", &max_files_str);
        sh.set_var("MAX_MINUTES", &max_minutes_str);

        let mut args = vec![
            "apply",
            "-f",
            ".dstack/benchmark.yml",
            "-n",
            &task_name,
            "-y",
            "--detach",
        ];

        if reuse {
            args.push("-R");
        }

        println!("  {task_name} ({ds_id})");
        cmd!(sh, "dstack {args...}").run()?;
        submitted.push(task_name);
    }

    println!();
    println!("All {} tasks submitted. Monitor with:", submitted.len());
    println!("  cargo xtask dstack ps");
    for task_name in &submitted {
        println!("  cargo xtask dstack logs {task_name}");
    }

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
    cmd!(sh, "s5cmd --endpoint-url {endpoint} cp {src} {local_dir}/").run()?;
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
