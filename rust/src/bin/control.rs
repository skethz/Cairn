//! Offline orchestration: fixed NVIDIA inventory utility and sibling launcher.
#[path = "../isolation.rs"]
mod isolation;
use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    Status {
        directory: PathBuf,
    },
    Queue {
        directory: PathBuf,
        #[arg(value_parser=clap::value_parser!(u8).range(4..=7))]
        gpu: u8,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        optimized: bool,
        #[arg(long, default_value_t=0, value_parser=clap::value_parser!(u8).range(0..=32))]
        retained_blocks: u8,
        #[arg(long)]
        deterministic: bool,
        #[arg(long)]
        fused_scan: bool,
        /// Wait for this run's training, final evaluation, and controller to finish.
        #[arg(long)]
        after_run: Option<String>,
        #[arg(required = true)]
        configs: Vec<PathBuf>,
    },
}
struct Lock(PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn bounded_json(p: &std::path::Path) -> Result<Value> {
    ensure!(
        p.is_file() && !p.is_symlink() && p.metadata()?.len() <= 2 * 1024 * 1024,
        "Expected bounded local JSON"
    );
    Ok(serde_json::from_slice(&std::fs::read(p)?)?)
}
fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
fn acquire_lock(path: PathBuf) -> Result<Lock> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(&path)?;
    write!(file, "{}", std::process::id())?;
    Ok(Lock(path))
}
fn write_record(path: &std::path::Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
fn retained_for(family: Option<&str>, requested: u8) -> u8 {
    match family {
        Some("full_mla") => requested.min(4),
        Some("dense_gqa") => requested.min(8),
        Some("kda_mla_operator") => requested.min(5),
        _ => requested,
    }
}
fn predecessor_ready(directory: &std::path::Path, gpu: u8, name: &str) -> Result<bool> {
    ensure!(safe_name(name), "Unsafe predecessor run name");
    let run = directory.join(name);
    let state = bounded_json(&run.join("status.json"))?;
    let run_lock = run.join("run.lock");
    let queue_lock = directory.join(format!("queue_gpu{gpu}.lock"));
    if state["status"] == "completed" {
        return Ok(!run_lock.exists() && !queue_lock.exists());
    }
    ensure!(state["status"] == "running", "Predecessor did not complete: {}", state["status"]);
    let alive = match std::fs::read_to_string(&run_lock) {
        Ok(value) => {
            let pid: i32 = value.trim().parse()?;
            ensure!(pid > 0, "Invalid predecessor PID");
            #[cfg(target_os = "linux")]
            let alive = unsafe { libc::kill(pid, 0) } == 0;
            #[cfg(not(target_os = "linux"))]
            let alive = true;
            alive
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(error.into()),
    };
    if alive { return Ok(false); }
    // Completion can occur between reading status and observing its process.
    ensure!(bounded_json(&run.join("status.json"))?["status"] == "completed",
        "Predecessor stopped without completing; successors remain unstarted");
    Ok(!run_lock.exists() && !queue_lock.exists())
}
fn main() -> Result<()> {
    isolation::activate(false)?;
    match Args::parse().command {
        Action::Status { directory } => {
            let mut runs = Vec::new();
            for e in std::fs::read_dir(directory)? {
                let e = e?;
                if !e.file_type()?.is_dir() {
                    continue;
                }
                let status = e.path().join("status.json");
                if status.exists() {
                    runs.push(json!({"run":e.file_name().to_string_lossy(),"status":bounded_json(&status)?}));
                }
            }
            println!("{}", serde_json::to_string_pretty(&runs)?);
        }
        Action::Queue {
            directory,
            gpu,
            configs,
            resume,
            optimized,
            retained_blocks,
            deterministic,
            fused_scan,
            after_run,
        } => {
            std::fs::create_dir_all(&directory)?;
            let pending_record = directory.join(format!("queue_gpu{gpu}.pending.json"));
            let pending_guard = if after_run.is_some() {
                Some(acquire_lock(directory.join(format!("queue_gpu{gpu}.pending.lock")))?)
            } else {
                None
            };
            let mut planned = Vec::new();
            for path in configs {
                let config = path.canonicalize()?;
                let c = bounded_json(&config)?;
                let name = c["run_name"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("Missing run name"))?;
                ensure!(
                    safe_name(name),
                    "Unsafe run name"
                );
                ensure!(after_run.as_deref() != Some(name), "A run cannot wait for itself");
                let output = directory.join(name);
                let continuing=output.exists();
                if continuing {
                    ensure!(resume && output.join("latest.json").is_file(),"Existing run requires --resume and a completed checkpoint");
                    ensure!(!output.join("run.lock").exists(),"Run lock exists; inspect its owner before resuming");
                }
                // A larger budget for the current hybrid must not silently
                // change the previously verified settings of queued models.
                let retained=retained_for(c["model"]["family"].as_str(), retained_blocks);
                planned.push((config, output, continuing, retained));
            }
            let launcher = std::env::current_exe()?
                .parent()
                .unwrap()
                .join("cairn-run")
                .canonicalize()?;
            let schedule: Vec<_> = planned.iter().map(|(config, output, continuing, retained)| {
                json!({"config":config,"run":output.file_name().unwrap().to_string_lossy(),
                    "resume":continuing,"retained_blocks":retained})
            }).collect();
            let mut pending = json!({"gpu":gpu,"pid":std::process::id(),
                "status":"waiting_for_predecessor","after_run":after_run,"planned":schedule});
            if let Some(name) = &after_run {
                write_record(&pending_record, &pending)?;
                loop {
                    match predecessor_ready(&directory, gpu, name) {
                        Ok(true) => break,
                        Ok(false) => std::thread::sleep(std::time::Duration::from_secs(10)),
                        Err(error) => {
                            pending["status"] = json!("blocked");
                            pending["error"] = json!(error.to_string());
                            write_record(&pending_record, &pending)?;
                            return Err(error);
                        }
                    }
                }
            }
            let _guard = acquire_lock(directory.join(format!("queue_gpu{gpu}.lock")))?;
            if after_run.is_some() {
                pending["status"] = json!("activated");
                write_record(&pending_record, &pending)?;
            }
            drop(pending_guard);
            let record=directory.join(format!("queue_gpu{gpu}.json"));
            let mut finished=Vec::<String>::new();
            for (config, output, continuing, retained) in planned {
                ensure!(output.exists() == continuing, "Queued run directory changed while waiting");
                ensure!(!output.join("run.lock").exists(), "Queued run is already active");
                ensure!(!output.join("stop.request").exists(), "Queued run has a stop request");
                let inventory = Command::new("/usr/bin/nvidia-smi")
                    .args([
                        "--query-gpu=uuid",
                        "--format=csv,noheader",
                        "--id",
                        &gpu.to_string(),
                    ])
                    .output()?;
                ensure!(
                    inventory.status.success() && inventory.stdout.len() < 1024,
                    "Cannot verify GPU availability"
                );
                let uuid = std::str::from_utf8(&inventory.stdout)?.trim();
                ensure!(uuid.starts_with("GPU-"), "Unexpected GPU identity");
                let processes = Command::new("/usr/bin/nvidia-smi")
                    .args(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
                    .output()?;
                ensure!(
                    processes.status.success() && processes.stdout.len() < 65536,
                    "Cannot inspect GPU occupants"
                );
                let occupied = std::str::from_utf8(&processes.stdout)?
                    .lines()
                    .any(|line| line.split(',').next().is_some_and(|id| id.trim() == uuid));
                ensure!(
                    !occupied,
                    "GPU {gpu} is already in use; refusing to start another run"
                );
                let log = OpenOptions::new()
                    .write(true)
                    .append(continuing)
                    .create(continuing)
                    .create_new(!continuing)
                    .open(output.with_extension("console.log"))?;
                let mut command=Command::new(&launcher);
                command.arg(gpu.to_string());
                if optimized { command.arg("--optimized"); }
                if deterministic { command.arg("--deterministic"); }
                if fused_scan { command.arg("--fused-scan"); }
                command.arg("--retained-blocks").arg(retained.to_string());
                command
                    .arg("train")
                    .arg(&config)
                    .arg(&output)
                    .stdin(Stdio::null())
                    .stdout(Stdio::from(log.try_clone()?))
                    .stderr(Stdio::from(log));
                if continuing { command.arg("--resume"); }
                let name=output.file_name().unwrap().to_string_lossy().to_string();
                write_record(&record,&json!({"gpu":gpu,"status":"running","current":name,"finished":finished,"planned":schedule,"resume":continuing,"optimized":optimized,"retained_blocks":retained,"deterministic":deterministic,"fused_scan":fused_scan}))?;
                let status=command.status()?;
                ensure!(
                    status.success(),
                    "Queued run failed: {}; later runs remain unstarted",
                    output.display()
                );
                let result=bounded_json(&output.join("status.json"))?;
                ensure!(result["status"]=="completed","Run stopped at a checkpoint; queued runs remain unstarted");
                finished.push(name);
            }
            write_record(&record,&json!({"gpu":gpu,"status":"completed","finished":finished}))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("cairn-queue-test-{}-{label}", std::process::id()));
        std::fs::create_dir_all(path.join("predecessor")).unwrap();
        path
    }
    #[test]
    fn waits_for_completion_and_both_locks() {
        let root = fixture("completion");
        let run = root.join("predecessor");
        write_record(&run.join("status.json"), &json!({"status":"running"})).unwrap();
        let worker = acquire_lock(run.join("run.lock")).unwrap();
        let controller = acquire_lock(root.join("queue_gpu4.lock")).unwrap();
        assert!(!predecessor_ready(&root, 4, "predecessor").unwrap());
        write_record(&run.join("status.json"), &json!({"status":"completed"})).unwrap();
        assert!(!predecessor_ready(&root, 4, "predecessor").unwrap());
        drop(worker);
        assert!(!predecessor_ready(&root, 4, "predecessor").unwrap());
        drop(controller);
        assert!(predecessor_ready(&root, 4, "predecessor").unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn paused_or_failed_predecessor_does_not_start_successors() {
        let root = fixture("failure");
        let state = root.join("predecessor/status.json");
        write_record(&state, &json!({"status":"stopped_at_checkpoint"})).unwrap();
        assert!(predecessor_ready(&root, 4, "predecessor").is_err());
        write_record(&state, &json!({"status":"running"})).unwrap();
        assert!(predecessor_ready(&root, 4, "predecessor").is_err());
        assert!(predecessor_ready(&root, 4, "../predecessor").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn each_family_keeps_its_verified_memory_budget() {
        for (family, expected) in [("cairn",14),("gdn_h_recipe_proxy",14),
            ("kda_mla_operator",5),("gdn_gqa",14),("dense_gqa",8),("full_mla",4)] {
            assert_eq!(retained_for(Some(family),14), expected);
            assert_eq!(retained_for(Some(family),2), 2);
        }
    }
}
