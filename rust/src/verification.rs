//! Reproducible cross-language evidence, never called by the training loop.
use crate::{
    benchmarks,
    data::Documents,
    io,
    model::{Config, Model},
    native, recurrence, tokenizer,
};
use anyhow::{Result, ensure};
use serde_json::json;
use std::{collections::HashMap, path::Path};
use tch::{Device, Kind, Tensor};
pub fn routing(output: &Path, device: Device) -> Result<()> {
    ensure!(device.is_cuda(), "Routing check requires CUDA");
    let mut cases=Vec::new();
    for (rows,experts,k,skew) in [(1,2,1,false),(259,16,2,false),(65536,16,2,false),(1025,64,1,true),(131,4,4,false)] {
        let selected=if skew { Tensor::zeros([rows,k],(Kind::Int64,device)) } else {
            Tensor::rand([rows,experts],(Kind::Float,device)).topk(k,-1,true,true).1
        };
        let (routes,counts)=native::routes(&selected,experts)?;
        ensure!(counts.iter().sum::<i64>()==rows*k,"Missing assignments");
        for e in 0..experts {
            let expected=selected.eq(e).nonzero();
            let observed=routes.select(0,e).narrow(0,0,counts[e as usize]);
            ensure!(observed.equal(&(expected.select(1,0)*k+expected.select(1,1))),"Stable route order mismatch");
        }
        cases.push(json!({"rows":rows,"experts":experts,"top_k":k,"skew":skew,"exact":true}));
    }
    io::atomic_json(output,&json!({"status":"passed","cases":cases}))
}
pub fn forward(
    config: &Path,
    weights: &Path,
    inputs: &Path,
    output: &Path,
    device: Device,
    backward: bool,
) -> Result<()> {
    let c = io::json(config, 100000)?;
    let cfg: Config = serde_json::from_value(c.get("model").unwrap_or(&c).clone())?;
    let mut m = Model::new(cfg, 17, device)?;
    m.load(io::tensors(weights, 8 * 1024_u64.pow(3))?)?;
    let ts: HashMap<_, _> = io::tensors(inputs, 128 * 1024 * 1024)?
        .into_iter()
        .map(|(n, t)| (n, t.to_device(device)))
        .collect();
    let ids = ts
        .get("ids")
        .ok_or_else(|| anyhow::anyhow!("Missing IDs"))?;
    let labels = ts
        .get("labels")
        .ok_or_else(|| anyhow::anyhow!("Missing labels"))?;
    ensure!(
        ids.kind() == Kind::Int64
            && ids.dim() == 2
            && labels.size() == ids.size()
            && ids.size()[1] <= 4096,
        "Invalid verification inputs"
    );
    let (logits, aux, load) =
        tch::no_grad(|| native::autocast(device.is_cuda(), || m.logits(ids, &labels.ne(-100))));
    let mut result = vec![
        ("logits".into(), logits.to_device(Device::Cpu)),
        ("aux".into(), aux.to_device(Device::Cpu)),
        ("load".into(), load.to_device(Device::Cpu)),
    ];
    let count = labels.ne(-100).sum(Kind::Int64).int64_value(&[]);
    if backward {
        let (loss, aux, _) = m.backward_recompute(ids, labels, count, 0.01, device.is_cuda())?;
        result.push(("loss".into(), Tensor::from_slice(&[loss, aux])));
        for p in &m.params {
            if p.tensor.grad().defined() {
                result.push((
                    format!("grad::{}", p.name),
                    p.tensor.grad().to_device(Device::Cpu),
                ));
            }
        }
    }
    Tensor::write_safetensors(&result, output)?;
    println!(
        "{}",
        json!({"parameters":Model::parameter_count(&m.config)?,"tensors":result.len(),"backward":backward})
    );
    Ok(())
}
pub fn protocol(data: &Path, fixture: &Path, output: &Path) -> Result<()> {
    let meta = io::json(&fixture.join("protocol_case.json"), 1024 * 1024)?;
    let mut result = Vec::new();
    let mut cursors = Vec::new();
    for (case, c) in meta["batches"].as_array().unwrap().iter().enumerate() {
        let mut dataset = Documents::new(
            data,
            "train",
            c["seed"].as_u64().unwrap(),
            c["context"].as_u64().unwrap() as usize,
            None,
        )?;
        dataset.cursor = c["cursor"].as_u64().unwrap();
        for batch in 0..2 {
            let (ids, labels) =
                dataset.batch(c["microbatch"].as_u64().unwrap() as usize, Device::Cpu)?;
            result.push((format!("ids::{case}::{batch}"), ids));
            result.push((format!("labels::{case}::{batch}"), labels));
        }
        cursors.push(dataset.cursor);
    }
    let t = tokenizer::load(&data.join("tokenizer.json"))?;
    let texts: Vec<String> = serde_json::from_value(meta["texts"].clone())?;
    let ids: Vec<_> = texts
        .iter()
        .map(|s| tokenizer::encode(&t, s))
        .collect::<Result<_>>()?;
    let byte = tokenizer::ByteBpe::train(&texts, 300, 2)?;
    let byte_ids: Vec<_> = texts
        .iter()
        .map(|s| byte.encode(s, true, true))
        .collect::<Result<_>>()?;
    let decoded: Vec<_> = byte_ids
        .iter()
        .map(|s| byte.decode(s, true, "strict"))
        .collect::<Result<_>>()?;
    let tokenizer = tokenizer::load(&fixture.join("choice_tokenizer.json"))?;
    for (i, (input, labels, _)) in benchmarks::choices(&tokenizer, &meta["choice"], 4, Device::Cpu)?
        .into_iter()
        .enumerate()
    {
        result.push((format!("choice_ids::{i}"), input));
        result.push((format!("choice_labels::{i}"), labels));
    }
    std::fs::create_dir_all(output)?;
    Tensor::write_safetensors(&result, output.join("actual.safetensors"))?;
    byte.save(&output.join("byte_bpe.json"))?;
    io::atomic_json(
        &output.join("actual.json"),
        &json!({"cursors":cursors,"token_ids":ids,"byte_merges":byte.merges,"byte_ids":byte_ids,"decoded":decoded}),
    )?;
    Ok(())
}
pub fn recurrence(fixture: &Path, output: &Path, device: Device) -> Result<()> {
    let mut out = Vec::new();
    for channels in [1, 4] {
        let inputs: HashMap<_, _> = io::tensors(
            &fixture.join(format!("recurrence_{channels}.safetensors")),
            1024 * 1024,
        )?
        .into_iter()
        .collect();
        for chunk in [1, 4, 8, 16] {
            let xs: Vec<_> = ["q", "k", "v", "decay", "beta", "initial"]
                .iter()
                .map(|n| {
                    inputs[*n]
                        .to_device(device)
                        .detach()
                        .set_requires_grad(true)
                })
                .collect();
            let (values, state) = recurrence::chunk_delta(
                &xs[0],
                &xs[1],
                &xs[2],
                &xs[3],
                &xs[4],
                chunk,
                Some(&xs[5]),
            );
            (&values.square().sum(None) + state.square().sum(None)).backward();
            let prefix = format!("{channels}::{chunk}");
            out.push((
                format!("out::{prefix}"),
                values.detach().to_device(Device::Cpu),
            ));
            out.push((
                format!("state::{prefix}"),
                state.detach().to_device(Device::Cpu),
            ));
            for (i, x) in xs.iter().enumerate() {
                out.push((
                    format!("grad::{prefix}::{i}"),
                    x.grad().to_device(Device::Cpu),
                ));
            }
        }
    }
    let out: Vec<_> = out.into_iter().map(|(n, t)| (n, t.contiguous())).collect();
    Tensor::write_safetensors(&out, output)?;
    Ok(())
}
pub fn timing(config: &Path, output: &Path, device: Device) -> Result<()> {
    let c = io::json(config, 100000)?;
    let cfg: Config = serde_json::from_value(c.get("model").unwrap_or(&c).clone())?;
    ensure!(
        Model::parameter_count(&cfg)? < 10_000_000,
        "Timing pilot is bounded to a small model"
    );
    let m = Model::new(cfg, 17, device)?;
    let opt = crate::optimizer::AdamW::new(&m);
    let ids =
        (Tensor::arange(256, (Kind::Int64, device)).reshape([2, 128]) * 29 + 11).remainder(256) + 3;
    let labels = (&ids * 11 + 7).remainder(m.config.vocab_size);
    let mut seconds = Vec::new();
    let mut losses = Vec::new();
    for i in 0..12 {
        if device.is_cuda() {
            tch::Cuda::synchronize(0);
        }
        let start = std::time::Instant::now();
        m.zero_grad()?;
        let (loss, _, _) = m.backward_recompute(&ids, &labels, 256, 0.01, device.is_cuda())?;
        opt.clip(&m, 1.)?;
        opt.step(&m, 4e-4)?;
        if device.is_cuda() {
            tch::Cuda::synchronize(0);
        }
        if i >= 2 {
            seconds.push(start.elapsed().as_secs_f64());
            losses.push(loss / 256.);
        }
    }
    let mut ordered = seconds.clone();
    ordered.sort_by(f64::total_cmp);
    let median = (ordered[4] + ordered[5]) / 2.;
    io::atomic_json(
        output,
        &json!({"implementation":"Rust block recomputation + native LibTorch 2.11.0","device":format!("{device:?}"),"threads":8,"parameters":Model::parameter_count(&m.config)?,"shape":[2,128],"warmup_updates":2,"measured_updates":10,"seconds":seconds,"median_seconds":median,"median_tokens_per_second":256./median,"nll":losses,"scope":"small-model pilot on shared host; not an isolated 1.3B GPU speed claim"}),
    )?;
    Ok(())
}
