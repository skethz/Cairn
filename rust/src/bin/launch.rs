//! Apply denial before the dynamic tensor libraries load.
#[path = "../isolation.rs"]
mod isolation;
use anyhow::{Result, ensure};
use std::{
    os::unix::process::CommandExt,
    process::{Command, Stdio},
};
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let device = args.next().unwrap_or_default();
    ensure!(
        ["cpu", "4", "5", "6", "7"].contains(&device.as_str()),
        "Usage: cairn-run cpu|4|5|6|7 <command and arguments>"
    );
    let exe = std::env::current_exe()?
        .parent()
        .unwrap()
        .join("cairn")
        .canonicalize()?;
    let temporary = std::env::temp_dir().join(format!("cairn-rust-{}", std::process::id()));
    std::fs::create_dir(&temporary)?;
    let mut command = Command::new(exe);
    command.env_clear().stdin(Stdio::null());
    command
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C.UTF-8")
        .env("TMPDIR", &temporary)
        .env("XDG_CACHE_HOME", &temporary)
        .env(
            "CUDA_VISIBLE_DEVICES",
            if device == "cpu" { "" } else { &device },
        )
        .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        .env("CUDA_MPS_PIPE_DIRECTORY", temporary.join("no-mps"))
        .env("OMP_NUM_THREADS", "8")
        .env("MKL_NUM_THREADS", "8")
        .env("TOKENIZERS_PARALLELISM", "false")
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .env("DO_NOT_TRACK", "1")
        .env("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        .arg("--device")
        .arg(if device == "cpu" { "cpu" } else { "cuda" })
        .args(args);
    unsafe {
        command.pre_exec(|| {
            #[cfg(target_os = "linux")]
            {
                if libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, 0u32) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            isolation::activate(false).map_err(std::io::Error::other)?;
            Ok(())
        });
    }
    let status = command.status();
    let _ = std::fs::remove_dir_all(&temporary);
    let status = status?;
    ensure!(status.success(), "Cairn exited with {status}");
    Ok(())
}
