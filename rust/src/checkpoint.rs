use crate::{io, model::Model, native, optimizer::AdamW};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};
use tch::{Device, Kind, Tensor};
pub fn save(dir: &Path, m: &Model, opt: &AdamW, state: &Value) -> Result<PathBuf> {
    let step = state["step"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("Missing step"))?;
    let name = format!("checkpoint_{step:08}");
    let target = dir.join(&name);
    let temporary = dir.join(format!("{name}.partial"));
    ensure!(
        !target.exists() && !temporary.exists(),
        "Checkpoint already exists"
    );
    std::fs::create_dir(&temporary)?;
    let weights: Vec<_> = m
        .params
        .iter()
        .map(|p| {
            (
                p.name.clone(),
                p.tensor.detach().to_device(Device::Cpu).contiguous(),
            )
        })
        .collect();
    Tensor::write_safetensors(&weights, temporary.join("model.safetensors"))?;
    drop(weights);
    let mut states = Vec::new();
    for (p, s) in m.params.iter().zip(&opt.states) {
        for (field, t) in [
            ("step", &s.step),
            ("exp_avg", &s.mean),
            ("exp_avg_sq", &s.variance),
        ] {
            states.push((
                format!("{}::{field}", p.name),
                t.detach().to_device(Device::Cpu).contiguous(),
            ));
        }
    }
    states.push(("__rng_cpu".into(), native::rng_get(false)?));
    if m.params[0].tensor.device().is_cuda() {
        states.push(("__rng_cuda".into(), native::rng_get(true)?));
    }
    Tensor::write_safetensors(&states, temporary.join("optimizer.safetensors"))?;
    drop(states);
    let mut meta = state.clone();
    meta["sha256"] = json!({"model.safetensors":io::hash(&temporary.join("model.safetensors"))?,"optimizer.safetensors":io::hash(&temporary.join("optimizer.safetensors"))?});
    io::atomic_json(&temporary.join("state.json"), &meta)?;
    std::fs::rename(&temporary, &target)?;
    io::atomic_json(&dir.join("latest.json"), &json!({"checkpoint":name}))?;
    let mut old = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let n = e.file_name().to_string_lossy().into_owned();
        if e.file_type()?.is_dir()
            && n.strip_prefix("checkpoint_")
                .is_some_and(|x| !x.is_empty() && x.bytes().all(|b| b.is_ascii_digit()))
        {
            old.push(e.path());
        }
    }
    old.sort();
    let n = old.len().saturating_sub(2);
    for p in old.into_iter().take(n) {
        std::fs::remove_dir_all(p)?;
    }
    Ok(target)
}
pub fn load(
    dir: &Path,
    m: &mut Model,
    opt: &mut AdamW,
    config_sha: &str,
    data_sha: &str,
) -> Result<Value> {
    let pointer = io::json(&dir.join("latest.json"), 4096)?;
    let name = pointer["checkpoint"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid checkpoint pointer"))?;
    ensure!(
        name.strip_prefix("checkpoint_")
            .is_some_and(|x| !x.is_empty() && x.bytes().all(|b| b.is_ascii_digit())),
        "Unsafe checkpoint name"
    );
    let target = dir.join(name);
    let state = io::json(&target.join("state.json"), 2 * 1024 * 1024)?;
    ensure!(
        state["config_sha256"] == config_sha && state["data_sha256"] == data_sha,
        "Checkpoint protocol mismatch"
    );
    for name in ["model.safetensors", "optimizer.safetensors"] {
        ensure!(
            io::hash(&target.join(name))? == state["sha256"][name].as_str().unwrap_or(""),
            "Checkpoint hash mismatch"
        );
    }
    m.load(io::tensors(
        &target.join("model.safetensors"),
        8 * 1024_u64.pow(3),
    )?)?;
    let names: HashMap<_, _> = m
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.clone(), i))
        .collect();
    let mut seen = HashSet::new();
    let device = m.params[0].tensor.device();
    let mut cpu_rng = None;
    let mut cuda_rng = None;
    tch::no_grad(|| -> Result<()> {
        for (name, t) in io::tensors(&target.join("optimizer.safetensors"), 24 * 1024_u64.pow(3))? {
            ensure!(seen.insert(name.clone()), "Duplicate optimizer field");
            if name == "__rng_cpu" {
                cpu_rng = Some(t);
                continue;
            }
            if name == "__rng_cuda" {
                cuda_rng = Some(t);
                continue;
            }
            let (p, field) = name
                .rsplit_once("::")
                .ok_or_else(|| anyhow::anyhow!("Invalid optimizer key"))?;
            let i = *names
                .get(p)
                .ok_or_else(|| anyhow::anyhow!("Unknown optimizer parameter"))?;
            let dst = match field {
                "step" => &mut opt.states[i].step,
                "exp_avg" => &mut opt.states[i].mean,
                "exp_avg_sq" => &mut opt.states[i].variance,
                _ => anyhow::bail!("Unknown optimizer field"),
            };
            ensure!(
                t.kind() == Kind::Float && t.size() == dst.size(),
                "Optimizer dtype/shape mismatch"
            );
            dst.copy_(&t.to_device(device));
        }
        Ok(())
    })?;
    for p in &m.params {
        for field in ["step", "exp_avg", "exp_avg_sq"] {
            ensure!(
                seen.contains(&format!("{}::{field}", p.name)),
                "Incomplete optimizer checkpoint"
            );
        }
    }
    native::rng_set(
        false,
        &cpu_rng.ok_or_else(|| anyhow::anyhow!("Missing CPU RNG"))?,
    )?;
    if device.is_cuda() {
        native::rng_set(
            true,
            &cuda_rng.ok_or_else(|| anyhow::anyhow!("Missing CUDA RNG"))?,
        )?;
    }
    Ok(state)
}
