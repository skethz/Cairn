mod benchmarks;
mod checkpoint;
mod data;
mod io;
mod isolation;
mod model;
mod native;
mod optimizer;
mod packed;
mod prepare;
mod recurrence;
mod tokenizer;
mod trainer;
mod verification;
use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tch::{Device, Kind, Tensor};
#[derive(Parser)]
struct Args {
    /// Enable validated execution optimizations without changing the model protocol.
    #[arg(long, global=true)]
    optimized: bool,
    /// Retain this many block graphs, trading memory for less recomputation.
    #[arg(long, global=true, default_value_t=0, value_parser=clap::value_parser!(u8).range(0..=32))]
    retained_blocks: u8,
    #[arg(long, global=true)]
    deterministic: bool,
    /// Experimental sequential recurrence VJP; validate rounding separately.
    #[arg(long, global=true)]
    fused_scan: bool,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    RoutingCheck { output: PathBuf },
    Probe,
    Timing {
        config: PathBuf,
        output: PathBuf,
    },
    ForwardCheck {
        config: PathBuf,
        weights: PathBuf,
        inputs: PathBuf,
        output: PathBuf,
        #[arg(long)]
        backward: bool,
    },
    ProtocolCheck {
        data: PathBuf,
        fixture: PathBuf,
        output: PathBuf,
    },
    RecurrenceCheck {
        fixture: PathBuf,
        output: PathBuf,
    },
    PrepareData {
        root: PathBuf,
        output: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },
    PrepareBenchmarks {
        root: PathBuf,
        output: PathBuf,
    },
    PackedInfer {
        config: PathBuf,
        weights: PathBuf,
        inputs: PathBuf,
        output: PathBuf,
        #[arg(long)]
        draft: bool,
    },
    Train {
        config: PathBuf,
        run_dir: PathBuf,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        steps: Option<u64>,
        #[arg(long)]
        profile: bool,
        #[arg(long)]
        save_profile: bool,
        #[arg(long)]
        evaluate: bool,
    },
    Tokenize {
        tokenizer: PathBuf,
        text: String,
        #[arg(long)]
        byte_bpe: bool,
    },
    Count {
        config: PathBuf,
    },
    Parity {
        fixture: PathBuf,
        output: PathBuf,
        #[arg(long)]
        recompute: bool,
        #[arg(long)]
        resume_check: bool,
    },
}
fn main() -> Result<()> {
    isolation::activate(false)?;
    let args = Args::parse();
    native::set_optimized(args.optimized);
    native::set_retained(args.retained_blocks as usize);
    native::set_scan(args.fused_scan);
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => {
            ensure!(
                ["4", "5", "6", "7"].contains(
                    &std::env::var("CUDA_VISIBLE_DEVICES")
                        .unwrap_or_default()
                        .as_str()
                ),
                "Explicit GPU4–7 restriction required"
            );
            ensure!(tch::Cuda::is_available(), "CUDA unavailable");
            let _ = Tensor::zeros([1], (Kind::Float, Device::Cuda(0)));
            Device::Cuda(0)
        }
        _ => anyhow::bail!("Device must be cpu or cuda"),
    };
    isolation::activate(true)?;
    native::settings()?;
    native::deterministic(args.deterministic)?;
    tch::set_num_threads(8);
    match args.command {
        Command::RoutingCheck { output } => verification::routing(&output, device)?,
        Command::Timing { config, output } => verification::timing(&config, &output, device)?,
        Command::ForwardCheck {
            config,
            weights,
            inputs,
            output,
            backward,
        } => verification::forward(&config, &weights, &inputs, &output, device, backward)?,
        Command::ProtocolCheck {
            data,
            fixture,
            output,
        } => verification::protocol(&data, &fixture, &output)?,
        Command::RecurrenceCheck { fixture, output } => {
            verification::recurrence(&fixture, &output, device)?
        }
        Command::PrepareData {
            root,
            output,
            tokenizer,
        } => prepare::data(&root, &output, tokenizer.as_deref())?,
        Command::PrepareBenchmarks { root, output } => prepare::benchmarks(&root, &output)?,
        Command::PackedInfer {
            config,
            weights,
            inputs,
            output,
            draft,
        } => {
            let c: packed::Config = serde_json::from_value(io::json(&config, 100000)?)?;
            let model = packed::Packed::load(c, &weights, device, 512 * 1024 * 1024)?;
            let input: std::collections::HashMap<_, _> = io::tensors(&inputs, 64 * 1024 * 1024)?
                .into_iter()
                .map(|(n, t)| (n, t.to_device(device)))
                .collect();
            let ids = input
                .get("ids")
                .ok_or_else(|| anyhow::anyhow!("Missing ids"))?;
            let out = tch::no_grad(|| {
                model.logits(ids, input.get("positions"), input.get("mask"), draft)
            })?;
            Tensor::write_safetensors(&[("logits", out.to_device(Device::Cpu))], output)?;
        }
        Command::Train {
            config,
            run_dir,
            resume,
            steps,
            profile,
            save_profile,
            evaluate,
        } => trainer::run(
            trainer::Options {
                config,
                directory: run_dir,
                resume,
                steps,
                profile,
                save_profile,
                evaluate,
            },
            device,
        )?,
        Command::Tokenize {
            tokenizer: path,
            text,
            byte_bpe,
        } => {
            if byte_bpe {
                println!(
                    "{}",
                    serde_json::to_string(
                        &tokenizer::ByteBpe::load(&path)?.encode(&text, false, false)?
                    )?
                );
            } else {
                println!(
                    "{}",
                    serde_json::to_string(&tokenizer::encode(&tokenizer::load(&path)?, &text)?)?
                );
            }
        }
        Command::Probe => {
            isolation::tight_negative_controls()?;
            let t = Tensor::ones([16, 16], (Kind::Float, device)).set_requires_grad(true);
            crate::native::autocast(device.is_cuda(), || t.matmul(&t).square().mean(None))
                .backward();
            println!(
                "{}",
                serde_json::json!({"device":format!("{device:?}"),"network":"denied by OS","finite_backward":t.grad().isfinite().all().int64_value(&[])==1,"python_runtime":false})
            );
        }
        Command::Count { config } => {
            let c = io::json(&config, 100000)?;
            let c: model::Config = serde_json::from_value(c.get("model").unwrap_or(&c).clone())?;
            println!("{}", model::Model::parameter_count(&c)?);
        }
        Command::Parity {
            fixture,
            output,
            recompute,
            resume_check,
        } => parity(&fixture, &output, device, recompute, resume_check)?,
    }
    Ok(())
}
fn parity(
    dir: &std::path::Path,
    out: &std::path::Path,
    device: Device,
    recompute: bool,
    resume_check: bool,
) -> Result<()> {
    let meta = io::json(&dir.join("case.json"), 100000)?;
    let c: model::Config = serde_json::from_value(meta["model"].clone())?;
    let mut m = model::Model::new(c, 17, device)?;
    let mut result = Vec::<(String, Tensor)>::new();
    for p in &m.params {
        result.push((
            format!("initialized::{}", p.name),
            p.tensor.detach().to_device(Device::Cpu).copy(),
        ));
    }
    m.load(io::tensors(
        &dir.join("weights.safetensors"),
        8 * 1024_u64.pow(3),
    )?)?;
    let inputs = io::tensors(&dir.join("inputs.safetensors"), 1024 * 1024 * 1024)?;
    let inputs: std::collections::HashMap<_, _> = inputs.into_iter().collect();
    let ids = inputs["ids"].to_device(device);
    let labels = inputs["labels"].to_device(device);
    let valid = labels.ne(-100);
    let bf16 = meta["bf16"].as_bool().unwrap_or(false);
    ensure!(!bf16 || device.is_cuda(), "BF16 parity uses CUDA");
    tch::no_grad(|| {
        crate::native::autocast(bf16, || {
            let mut x = m.embed(&ids);
            result.push(("embedding".into(), x.to_device(Device::Cpu).copy()));
            for i in 0..m.config.layers {
                let (y, aux, load) = m.block(i, &x, &valid);
                x = y;
                result.push((format!("block::{i}"), x.to_device(Device::Cpu).copy()));
                result.push((format!("aux::{i}"), aux.to_device(Device::Cpu).copy()));
                result.push((format!("load::{i}"), load.to_device(Device::Cpu).copy()));
            }
            let (l, a, load) = m.logits(&ids, &valid);
            result.push(("logits".into(), l.to_device(Device::Cpu)));
            result.push(("aux".into(), a.to_device(Device::Cpu)));
            result.push(("load".into(), load.to_device(Device::Cpu)));
        })
    });
    let mut opt = optimizer::AdamW::new(&m);
    std::fs::create_dir_all(out)?;
    let count = valid.sum(Kind::Int64).int64_value(&[]);
    for step in 0..3 {
        m.zero_grad()?;
        let (loss, aux) = if recompute {
            let (l, a, _) = m.backward_recompute(&ids, &labels, count, 0.01, bf16)?;
            (l, a)
        } else {
            let (loss, aux, _) = crate::native::autocast(bf16, || m.loss(&ids, &labels));
            let values = (loss.double_value(&[]), aux.double_value(&[]));
            (&loss / count + &aux * 0.01).backward();
            values
        };
        result.push((format!("loss::{step}"), Tensor::from_slice(&[loss, aux])));
        for p in &m.params {
            if p.tensor.grad().defined() {
                result.push((
                    format!("grad::{step}::{}", p.name),
                    p.tensor.grad().to_device(Device::Cpu).copy(),
                ));
            }
        }
        let norm = opt.clip(&m, 1.)?;
        result.push((
            format!("gradient_norm::{step}"),
            Tensor::from_slice(&[norm]),
        ));
        opt.step(&m, 4e-4)?;
        for (i, p) in m.params.iter().enumerate() {
            result.push((
                format!("updated::{step}::{}", p.name),
                p.tensor.detach().to_device(Device::Cpu).copy(),
            ));
            result.push((
                format!("mean::{step}::{}", p.name),
                opt.states[i].mean.to_device(Device::Cpu).copy(),
            ));
            result.push((
                format!("variance::{step}::{}", p.name),
                opt.states[i].variance.to_device(Device::Cpu).copy(),
            ));
        }
        if step == 0 && resume_check {
            let state = serde_json::json!({"step":1,"valid_tokens":count,"data_cursor":2,"config_sha256":"fixture-config","data_sha256":"fixture-data"});
            checkpoint::save(out, &m, &opt, &state)?;
            // Destroy in-memory state before restoring; subsequent updates must
            // match the uninterrupted Python trajectory, including moments.
            m = model::Model::new(m.config.clone(), 91, device)?;
            opt = optimizer::AdamW::new(&m);
            let restored =
                checkpoint::load(out, &mut m, &mut opt, "fixture-config", "fixture-data")?;
            ensure!(
                restored["data_cursor"] == 2 && restored["valid_tokens"] == count,
                "Cursor restore mismatch"
            );
        }
    }
    std::fs::create_dir_all(out)?;
    Tensor::write_safetensors(&result, out.join("actual.safetensors"))?;
    println!(
        "{}",
        serde_json::json!({"status":"computed","tensors":result.len(),"recompute":recompute,"device":format!("{device:?}")})
    );
    Ok(())
}
