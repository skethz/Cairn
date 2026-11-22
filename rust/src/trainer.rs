use crate::{
    benchmarks, checkpoint,
    data::{Documents, verify},
    io,
    model::{Config, Model},
    optimizer::AdamW,
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
use tch::{Device, Kind, Tensor};
pub struct Options {
    pub config: PathBuf,
    pub directory: PathBuf,
    pub resume: bool,
    pub steps: Option<u64>,
    pub evaluate: bool,
    pub profile: bool,
    pub save_profile: bool,
}
fn log(dir: &Path, event: Value) -> Result<()> {
    println!("{event}");
    let mut f = OpenOptions::new()
        .append(true)
        .create(true)
        .open(dir.join("metrics.jsonl"))?;
    writeln!(f, "{event}")?;
    Ok(())
}
fn u(v: &Value, key: &str) -> Result<u64> {
    v[key]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("Missing integer {key}"))
}
fn f(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("Missing number {key}"))
}
struct Lock(PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
pub fn run(o: Options, device: Device) -> Result<()> {
    ensure!(
        device.is_cuda(),
        "Training requires the explicitly selected GPU"
    );
    std::fs::create_dir_all(&o.directory)?;
    let lock = o.directory.join("run.lock");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)?;
    write!(file, "{}", std::process::id())?;
    let _lock = Lock(lock);
    let result = inner(&o, device);
    if let Err(error) = &result {
        io::atomic_json(
            &o.directory.join("status.json"),
            &json!({"status":"failed","message":error.to_string()}),
        )?;
    }
    result
}
fn evaluate(m: &Model, d: &mut Documents, batches: u64, micro: usize) -> Result<Value> {
    d.cursor = 0;
    let mut losses = 0.;
    let mut tokens = 0i64;
    let device = m.params[0].tensor.device();
    tch::no_grad(|| -> Result<()> {
        for _ in 0..batches {
            let (ids, labels) = d.batch(micro, device)?;
            let (loss, _, _) = crate::native::autocast(true, || m.loss(&ids, &labels));
            losses += loss.double_value(&[]);
            tokens += labels.ne(-100).sum(Kind::Int64).int64_value(&[]);
        }
        Ok(())
    })?;
    ensure!(tokens > 0 && losses.is_finite(), "Invalid evaluation loss");
    let nll = losses / tokens as f64;
    Ok(
        json!({"nll":nll,"perplexity":nll.min(50.).exp(),"scored_tokens":tokens,"records":batches*micro as u64}),
    )
}
fn final_evaluation(m: &Model, c: &Value, data: &Path, dir: &Path, step: u64) -> Result<()> {
    let mut test = Documents::new(
        data,
        "test",
        0,
        u(&c["training"], "context")? as usize,
        None,
    )?;
    let size = test.size;
    let mut result = evaluate(m, &mut test, size, 1)?;
    result["event"] = json!("held_out_test");
    result["step"] = json!(step);
    log(dir, result)?;
    if let Some(b) = c["benchmark_directory"].as_str() {
        let result = benchmarks::run(
            m,
            Path::new(b),
            data,
            u(&c["training"], "context")? as usize,
            &dir.join(format!("benchmark_step_{step:08}.json")),
            None,
        )?;
        log(
            dir,
            json!({"event":"benchmarks_complete","step":step,"results":result}),
        )?;
    }
    Ok(())
}
fn inner(o: &Options, device: Device) -> Result<()> {
    ensure!(!o.save_profile || o.profile, "--save-profile requires --profile");
    let dir = &o.directory;
    ensure!(
        !dir.join("metrics.jsonl").exists() || o.resume,
        "Existing history requires explicit resume"
    );
    let c = io::json(&o.config, 100000)?;
    let config_sha = io::hash(&o.config)?;
    let cfg: Config = serde_json::from_value(c["model"].clone())?;
    let parameters = Model::parameter_count(&cfg)?;
    ensure!(
        parameters
            >= c["parameter_count_interval"][0]
                .as_u64()
                .unwrap_or(u64::MAX)
            && parameters
                <= c["parameter_count_interval"][1]
                    .as_u64()
                    .unwrap_or(0)
                    .min(2_000_000_000),
        "Parameter count outside protocol"
    );
    let t = &c["training"];
    let context = u(t, "context")? as usize;
    let micro = u(t, "microbatch")? as usize;
    let accumulation = u(t, "accumulation")?;
    let budget = u(t, "valid_token_budget")?;
    let warmup = f(t, "warmup_fraction")?;
    let peak = f(t, "peak_lr")?;
    let minimum = f(t, "minimum_lr_fraction")?;
    let router = f(t, "router_coefficient")?;
    let clip = f(t, "clip")?;
    ensure!(
        (1..=4096).contains(&context)
            && (1..=16).contains(&micro)
            && (1..=128).contains(&accumulation)
            && budget > 0
            && warmup > 0.
            && warmup < 1.
            && peak > 0.
            && minimum >= 0.
            && minimum <= 1.
            && router >= 0.
            && clip > 0.,
        "Invalid training bounds"
    );
    let eval_every = u(t, "eval_every")?;
    let eval_batches = u(t, "eval_batches")?;
    let checkpoint_every = u(t, "checkpoint_every")?;
    ensure!(
        eval_every > 0 && eval_batches > 0 && checkpoint_every > 0,
        "Invalid intervals"
    );
    let data = Path::new(
        c["data_directory"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing data directory"))?,
    );
    let (manifest, data_sha) = verify(data)?;
    ensure!(
        manifest["tokenizer"]["vocab_size"].as_i64() == Some(cfg.vocab_size),
        "Tokenizer vocabulary mismatch"
    );
    let seed = u(t, "seed")?;
    tch::manual_seed(seed as i64);
    let mut model = Model::new(cfg, seed, device)?;
    let mut opt = AdamW::new(&model);
    let mut dataset = Documents::new(data, "train", seed, context, None)?;
    let mut dev = Documents::new(data, "validation", 0, context, None)?;
    let mut state = json!({"step":0,"valid_tokens":0,"data_cursor":0,"config_sha256":config_sha,"data_sha256":data_sha});
    if o.resume {
        state = checkpoint::load(dir, &mut model, &mut opt, &config_sha, &data_sha)?;
        dataset.cursor = u(&state, "data_cursor")?;
    }
    let exe = std::env::current_exe()?;
    io::atomic_json(
        &dir.join("provenance.json"),
        &json!({"language":"Rust","native_backend":"LibTorch 2.11.0; no Python runtime","executable_sha256":io::hash(&exe)?,"parameters":parameters,"config_sha256":config_sha,"data_sha256":data_sha,"config":c,"network":"mandatory seccomp denial before model creation","gpu_physical_index":std::env::var("CUDA_VISIBLE_DEVICES")?,"original_code_executed":false,"pretrained_weights_loaded":false,"optimizer":"native fused AdamW; FP32 parameters/gradients/moments; BF16 autocast","activation_recomputation":"Rust block VJP","optimized_execution":crate::native::optimized()}),
    )?;
    io::atomic_json(&dir.join("execution.json"), &json!({
        "optimized_routing":crate::native::optimized(),
        "fused_scan":crate::native::scan_enabled(),
        "retained_blocks":crate::native::retained(),
        "deterministic":crate::native::deterministic_enabled(),
        "config_sha256":config_sha,
        "data_sha256":data_sha,
        "starting_step":state["step"],
        "starting_valid_tokens":state["valid_tokens"]
    }))?;
    log(
        dir,
        json!({"event":if o.resume{"resumed"}else{"initialized"},"step":state["step"],"parameters":parameters}),
    )?;
    if o.evaluate {
        ensure!(o.resume, "Evaluation requires a checkpoint");
        let mut result = evaluate(&model, &mut dev, eval_batches, micro)?;
        result["event"] = json!("validation");
        log(dir, result)?;
        return final_evaluation(&model, &c, data, dir, u(&state, "step")?);
    }
    if u(&state, "step")? == 0 && !o.profile {
        let mut result = evaluate(&model, &mut dev, eval_batches, micro)?;
        result["event"] = json!("validation");
        result["step"] = json!(0);
        log(dir, result)?;
    }
    let max_steps = o.steps.unwrap_or(u(t, "max_steps")?);
    let started = Instant::now();
    let initial = u(&state, "valid_tokens")?;
    crate::native::memory(true)?;
    while u(&state, "step")? < max_steps && u(&state, "valid_tokens")? < budget {
        let start = Instant::now();
        let tokens = u(&state, "valid_tokens")?;
        let fraction = tokens as f64 / budget as f64;
        let multiplier = if fraction < warmup {
            minimum.max(fraction / warmup)
        } else {
            let phase = ((fraction - warmup) / (1. - warmup)).min(1.);
            0.1 + 0.9 * (1. + (std::f64::consts::PI * phase).cos()) / 2.
        };
        let lr = peak * multiplier;
        let mut batches = Vec::new();
        let mut valid_total = 0i64;
        let mut slots = 0usize;
        for _ in 0..accumulation {
            let (ids, labels) = dataset.batch(micro, device)?;
            let mut count = labels.ne(-100).sum(Kind::Int64).int64_value(&[]);
            let left = (budget - tokens) as i64 - valid_total;
            if count > left {
                let mut flat = labels.flatten(0, -1);
                let keep = flat.ne(-100).nonzero().select(1, 0);
                let _ = flat.index_fill_(0, &keep.narrow(0, left, count - left), -100);
                count = left;
            }
            if count > 0 {
                slots += ids.numel();
                batches.push((ids, labels, count));
                valid_total += count;
            }
            if valid_total as u64 >= budget - tokens {
                break;
            }
        }
        ensure!(valid_total > 0, "Empty update");
        model.zero_grad()?;
        let mut loss_total = 0.;
        let mut aux_total = 0.;
        let mut routing: Option<Tensor> = None;
        for (ids, labels, count) in &batches {
            let (loss, aux, load) = if model.config.activation_checkpointing {
                let router_scale =
                    Tensor::ones([], (Kind::Float, device)) / valid_total * (*count) * router;
                model.backward_recompute_scaled(ids, labels, valid_total, &router_scale, true)?
            } else {
                let (loss, aux, load) = crate::native::autocast(true, || model.loss(ids, labels));
                let objective = &loss / valid_total + &aux * router * (*count as f64) / valid_total;
                ensure!(
                    objective.isfinite().all().int64_value(&[]) == 1,
                    "Nonfinite objective"
                );
                objective.backward();
                (loss.double_value(&[]), aux.double_value(&[]), load)
            };
            ensure!(loss.is_finite() && aux.is_finite(), "Nonfinite loss");
            loss_total += loss;
            aux_total += aux * (*count as f64) / valid_total as f64;
            let weighted = load * (*count as f64) / valid_total;
            routing = Some(match routing {
                None => weighted,
                Some(previous) => previous + weighted,
            });
        }
        let norm = opt.clip(&model, clip)?;
        opt.step(&model, lr)?;
        tch::Cuda::synchronize(0);
        let step = u(&state, "step")? + 1;
        state["step"] = json!(step);
        state["valid_tokens"] = json!(tokens + valid_total as u64);
        state["data_cursor"] = json!(dataset.cursor);
        let seconds = start.elapsed().as_secs_f64();
        let routing: Vec<f32> = Vec::try_from(routing.unwrap().to_device(Device::Cpu))?;
        let (allocated,reserved)=crate::native::memory(false)?;
        let entry = json!({"event":"train","step":step,"valid_tokens":state["valid_tokens"],"batch_valid_tokens":valid_total,"nll":loss_total/valid_total as f64,"router_aux":aux_total,"lr":lr,"gradient_norm":norm,"seconds":seconds,"tokens_per_second":valid_total as f64/seconds,"peak_allocated_GiB":allocated,"peak_reserved_GiB":reserved,"retained_blocks":crate::native::retained(),"optimized_execution":crate::native::optimized(),"padding_fraction":1.-valid_total as f64/slots as f64,"unused_nominal_context_fraction":1.-valid_total as f64/(batches.len()*micro*context) as f64,"mean_expert_load":routing});
        log(dir, entry.clone())?;
        let mut status = state.clone();
        status["status"] = json!("running");
        status["last_step"] = entry;
        io::atomic_json(&dir.join("status.json"), &status)?;
        // Explicit local maintenance request: finish this update and atomically
        // checkpoint before releasing the GPU. Never truncate an optimizer step.
        if !o.profile && dir.join("stop.request").is_file() {
            if !dir.join(format!("checkpoint_{step:08}")).exists() {
                checkpoint::save(dir, &model, &opt, &state)?;
            }
            state["status"]=json!("stopped_at_checkpoint");
            io::atomic_json(&dir.join("status.json"), &state)?;
            log(dir, json!({"event":"stopped_at_checkpoint","step":step,"valid_tokens":state["valid_tokens"]}))?;
            return Ok(());
        }
        if !o.profile && step % eval_every == 0 {
            let mut result = evaluate(&model, &mut dev, eval_batches, micro)?;
            result["event"] = json!("validation");
            result["step"] = json!(step);
            log(dir, result)?;
        }
        if !o.profile
            && (step == t["initial_checkpoint_step"].as_u64().unwrap_or(10)
                || step % checkpoint_every == 0)
        {
            let path = checkpoint::save(dir, &model, &opt, &state)?;
            log(dir, json!({"event":"checkpoint","step":step,"path":path}))?;
        }
    }
    let step = u(&state, "step")?;
    if o.save_profile {
        checkpoint::save(dir, &model, &opt, &state)?;
        let gradients: Vec<_> = model.params.iter().filter(|p| p.tensor.grad().defined())
            .map(|p| (p.name.clone(), p.tensor.grad().to_device(Device::Cpu).contiguous())).collect();
        Tensor::write_safetensors(&gradients, dir.join("final_gradients.safetensors"))?;
    }
    if !o.profile {
        if !dir.join(format!("checkpoint_{step:08}")).exists() {
            checkpoint::save(dir, &model, &opt, &state)?;
        }
        let mut result = evaluate(&model, &mut dev, eval_batches, micro)?;
        result["event"] = json!("validation");
        result["step"] = json!(step);
        log(dir, result)?;
        final_evaluation(&model, &c, data, dir, step)?;
    }
    state["status"] = json!(if o.profile {
        "profile_complete"
    } else {
        "completed"
    });
    state["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    state["sustained_tokens_per_second"] =
        json!((u(&state, "valid_tokens")? - initial) as f64 / started.elapsed().as_secs_f64());
    io::atomic_json(&dir.join("status.json"), &state)?;
    log(dir, state)
}
