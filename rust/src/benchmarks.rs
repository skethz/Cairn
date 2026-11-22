use crate::{io, model::Model, tokenizer};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader, Write},
    path::Path,
};
use tch::{Device, Tensor};
pub fn choices(
    t: &tokenizers::Tokenizer,
    row: &Value,
    context: usize,
    device: Device,
) -> Result<Vec<(Tensor, Tensor, usize)>> {
    let prompt = tokenizer::encode(
        t,
        row["prompt"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing prompt"))?,
    )?;
    let choices: Vec<_> = row["choices"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Missing choices"))?
        .iter()
        .map(|x| tokenizer::encode(t, x.as_str().unwrap_or("")))
        .collect::<Result<_>>()?;
    ensure!(
        choices.len() >= 2
            && choices.len() <= 32
            && choices.iter().all(|x| !x.is_empty() && x.len() < context),
        "Unsupported candidate length"
    );
    let longest = choices.iter().map(|x| x.len()).max().unwrap();
    let prompt = &prompt[prompt.len().saturating_sub(context - longest)..];
    let mut results = Vec::new();
    for candidate in choices {
        let n = candidate.len();
        let mut all = vec![1i64];
        all.extend_from_slice(prompt);
        all.extend(candidate);
        let ids = Tensor::from_slice(&all[..all.len() - 1])
            .unsqueeze(0)
            .to_device(device);
        let mut labels = all[1..].to_vec();
        labels[..prompt.len()].fill(-100);
        results.push((
            ids,
            Tensor::from_slice(&labels).unsqueeze(0).to_device(device),
            n,
        ));
    }
    Ok(results)
}
pub fn run(
    model: &Model,
    dir: &Path,
    data: &Path,
    context: usize,
    output: &Path,
    limit: Option<usize>,
) -> Result<Value> {
    let m = io::json(&dir.join("manifest.json"), 2 * 1024 * 1024)?;
    ensure!(
        m["status"] == "frozen" && m["protocol"] == "CairnLocal-v1",
        "Unsupported benchmark protocol"
    );
    let tokenizer = tokenizer::load(&data.join("tokenizer.json"))?;
    let device = model.params[0].tensor.device();
    let mut results = json!({"protocol":"CairnLocal-v1","manifest_sha256":io::hash(&dir.join("manifest.json"))?,"limit":limit,"tasks":{},"complete_benchmark_contamination_audit":false});
    let mut predictions = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.with_extension("predictions.jsonl"))?;
    let files = m["files"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Missing files"))?;
    let mut names: Vec<_> = files.keys().collect();
    names.sort();
    tch::no_grad(|| -> Result<()> {
        for name in names {
            ensure!(
                Path::new(name).file_name().and_then(|x| x.to_str()) == Some(name.as_str()),
                "Unsafe benchmark path"
            );
            let p = dir.join(name);
            ensure!(
                io::hash(&p)? == files[name]["sha256"].as_str().unwrap_or(""),
                "Benchmark hash mismatch"
            );
            let f = io::regular(&p, 256 * 1024 * 1024)?;
            ensure!(
                f.metadata()?.len() == files[name]["bytes"].as_u64().unwrap_or(0),
                "Benchmark size mismatch"
            );
            let (mut correct, mut normalized_correct, mut count) = (0, 0, 0);
            for line in BufReader::new(f).lines().take(limit.unwrap_or(usize::MAX)) {
                let row: Value = serde_json::from_str(&line?)?;
                let mut scores = Vec::new();
                let mut normalized = Vec::new();
                for (ids, labels, n) in choices(&tokenizer, &row, context, device)? {
                    let (loss, _, _) =
                        crate::native::autocast(device.is_cuda(), || model.loss(&ids, &labels));
                    let score = -loss.double_value(&[]);
                    ensure!(score.is_finite(), "Nonfinite benchmark loss");
                    scores.push(score);
                    normalized.push(score / n as f64);
                }
                let argmax = |xs: &[f64]| {
                    let mut i = 0;
                    for j in 1..xs.len() {
                        if xs[j] > xs[i] {
                            i = j;
                        }
                    }
                    i
                };
                let pred = argmax(&scores);
                let norm_pred = argmax(&normalized);
                let answer = row["answer"]
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("Missing gold choice"))?
                    as usize;
                ensure!(answer < scores.len(), "Invalid gold choice");
                correct += usize::from(pred == answer);
                normalized_correct += usize::from(norm_pred == answer);
                count += 1;
                writeln!(
                    predictions,
                    "{}",
                    json!({"task":name,"id":row["id"],"answer":answer,"sum_logprob":scores,"mean_logprob":normalized,"prediction":pred,"normalized_prediction":norm_pred})
                )?;
            }
            ensure!(count > 0, "Empty benchmark");
            let task = name.strip_suffix(".jsonl").unwrap_or(name);
            results["tasks"][task] = json!({"examples":count,"accuracy":correct as f64/count as f64,"token_normalized_accuracy":normalized_correct as f64/count as f64});
        }
        Ok(())
    })?;
    io::atomic_json(output, &results)?;
    Ok(results)
}
