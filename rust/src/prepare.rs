//! Offline preparation of already staged, hash-verified public data.
use crate::{io, tokenizer};
use anyhow::{Result, ensure};
use parquet::file::{reader::FileReader, serialized_reader::SerializedFileReader};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs::OpenOptions,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};
fn staged(root: &Path, r: &Value) -> Result<PathBuf> {
    let s = r["path"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing staged path"))?;
    ensure!(
        Path::new(s)
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_))),
        "Unsafe staged path"
    );
    let p = root.join(s);
    ensure!(
        io::hash(&p)? == r["sha256"].as_str().unwrap_or(""),
        "Staged file hash mismatch"
    );
    Ok(p)
}
fn parquet_rows(path: &Path, mut f: impl FnMut(Value) -> Result<bool>) -> Result<bool> {
    let reader = SerializedFileReader::new(io::regular(path, 16 * 1024_u64.pow(3))?)?;
    for row in reader.get_row_iter(None)? {
        if !f(row?.to_json_value())? {
            return Ok(false);
        }
    }
    Ok(true)
}
fn documents(
    root: &Path,
    manifest: &Value,
    mut f: impl FnMut(String) -> Result<bool>,
) -> Result<()> {
    for r in manifest["files"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Missing source files"))?
    {
        let path = staged(root, r)?;
        if !parquet_rows(&path, |row| {
            if let Some(text) = row["text"].as_str() {
                let chars = text.chars().count();
                if (128..=500000).contains(&chars) {
                    return f(text.to_owned());
                }
            }
            Ok(true)
        })? {
            break;
        }
    }
    Ok(())
}
fn whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(whitespace)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}
fn normalized_hash(text: &str) -> [u8; 32] {
    Sha256::digest(words(text).join(" ")).into()
}
fn split(d: &[u8; 32]) -> &'static str {
    let n = u64::from_be_bytes(d[..8].try_into().unwrap()) % 10000;
    if n < 20 {
        "test"
    } else if n < 40 {
        "validation"
    } else {
        "train"
    }
}
pub fn data(root: &Path, out: &Path, frozen_tokenizer: Option<&Path>) -> Result<()> {
    ensure!(
        !out.exists() || std::fs::read_dir(out)?.next().is_none(),
        "Output directory must be empty; frozen data cannot be overwritten"
    );
    std::fs::create_dir_all(out)?;
    let manifest = io::json(
        &root.join("evidence/download_manifest.json"),
        2 * 1024 * 1024,
    )?;
    let mut sample_bytes = 0usize;
    let mut tokenizer = if let Some(path) = frozen_tokenizer {
        tokenizer::load(path)?
    } else {
        let mut sample = Vec::new();
        let mut seen = HashSet::new();
        documents(root, &manifest, |text| {
            let hash = normalized_hash(&text);
            if split(&hash) != "train" || !seen.insert(hash) {
                return Ok(true);
            }
            sample_bytes += text.len();
            sample.push(text);
            Ok(sample_bytes < 64 * 1024 * 1024)
        })?;
        let mut t = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        t.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, true),
        ));
        t.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        let b = tokenizers::models::bpe::BpeTrainer::builder()
            .vocab_size(32000)
            .min_frequency(2)
            .show_progress(false)
            .initial_alphabet(
                tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
                    .into_iter()
                    .collect(),
            )
            .special_tokens(
                ["<pad>", "<bos>", "<eos>"]
                    .iter()
                    .map(|s| tokenizers::AddedToken::from(*s, true))
                    .collect(),
            )
            .build();
        let mut trainer: tokenizers::models::TrainerWrapper = b.into();
        t.train(&mut trainer, sample.into_iter())
            .map_err(|e| anyhow::anyhow!("BPE training: {e}"))?;
        t
    };
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
    ensure!(
        tokenizer.get_vocab_size(true) == 32000,
        "Expected 32000-token vocabulary"
    );
    if let Some(path) = frozen_tokenizer {
        std::fs::copy(path, out.join("tokenizer.json"))?;
    } else {
        tokenizer
            .save(out.join("tokenizer.json"), false)
            .map_err(|e| anyhow::anyhow!("Tokenizer save: {e}"))?;
    }
    struct Sink {
        tokens: BufWriter<std::fs::File>,
        index: BufWriter<std::fs::File>,
        offset: u64,
        docs: u64,
        records: u64,
        targets: u64,
    }
    let mut sinks = BTreeMap::new();
    for name in ["train", "validation", "test"] {
        let open = |suffix: &str| -> Result<_> {
            Ok(BufWriter::new(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(out.join(format!("{name}.{suffix}.partial")))?,
            ))
        };
        sinks.insert(
            name,
            Sink {
                tokens: open("tokens")?,
                index: open("index")?,
                offset: 0,
                docs: 0,
                records: 0,
                targets: 0,
            },
        );
    }
    let mut seen = HashSet::new();
    let mut edges = HashSet::new();
    let mut removed = 0u64;
    let mut pending = Vec::<(String, &'static str)>::new();
    let flush = |pending: &mut Vec<(String, &'static str)>,
                 sinks: &mut BTreeMap<&'static str, Sink>|
     -> Result<()> {
        let enc = tokenizer
            .encode_batch(pending.iter().map(|(s, _)| s.as_str()).collect(), false)
            .map_err(|e| anyhow::anyhow!("Encoding: {e}"))?;
        for ((_, name), encoding) in pending.iter().zip(enc) {
            let mut ids = vec![1u32];
            ids.extend_from_slice(encoding.get_ids());
            ids.push(2);
            if ids.len() < 34 {
                continue;
            }
            let sink = sinks.get_mut(name).unwrap();
            for &id in &ids {
                ensure!(id < 32000, "Out-of-range token");
                sink.tokens.write_all(&(id as u16).to_le_bytes())?;
            }
            for pos in (0..ids.len() - 1).step_by(4096) {
                let n = 4097.min(ids.len() - pos);
                sink.index
                    .write_all(&(sink.offset + pos as u64).to_le_bytes())?;
                sink.index.write_all(&(n as u64).to_le_bytes())?;
                sink.records += 1;
                sink.targets += (n - 1) as u64;
            }
            sink.offset += ids.len() as u64;
            sink.docs += 1;
        }
        pending.clear();
        Ok(())
    };
    documents(root, &manifest, |text| {
        let digest = normalized_hash(&text);
        let w = words(&text);
        let edge: [u8; 32] = Sha256::digest(format!(
            "{}\0{}",
            w[..w.len().min(64)].join(" "),
            w[w.len().saturating_sub(64)..].join(" ")
        ))
        .into();
        if seen.contains(&digest) || edges.contains(&edge) {
            removed += 1;
            return Ok(true);
        }
        seen.insert(digest);
        edges.insert(edge);
        pending.push((text, split(&digest)));
        if pending.len() >= 512 {
            flush(&mut pending, &mut sinks)?;
        }
        Ok(sinks["train"].targets < 1_250_000_000)
    })?;
    if !pending.is_empty() {
        flush(&mut pending, &mut sinks)?;
    }
    let mut counts = json!({});
    for (name, mut sink) in sinks {
        sink.tokens.flush()?;
        sink.index.flush()?;
        counts[name] = json!({"tokens":sink.offset,"documents":sink.docs,"records":sink.records,"target_tokens":sink.targets});
        drop(sink);
        for suffix in ["tokens", "index"] {
            std::fs::rename(
                out.join(format!("{name}.{suffix}.partial")),
                out.join(format!("{name}.{suffix}.bin")),
            )?;
        }
    }
    ensure!(
        counts["train"]["target_tokens"].as_u64().unwrap_or(0) >= 50_000_000
            && counts["validation"]["records"].as_u64().unwrap_or(0) >= 128
            && counts["test"]["records"].as_u64().unwrap_or(0) >= 128,
        "Insufficient prepared data"
    );
    let mut files = json!({});
    for entry in std::fs::read_dir(out)? {
        let p = entry?.path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        files[&name] = json!({"sha256":io::hash(&p)?,"bytes":p.metadata()?.len()});
    }
    io::atomic_json(
        &out.join("manifest.json"),
        &json!({"status":"frozen","source":manifest,"tokenizer":{"vocab_size":32000,"training_only":true,"sample_bytes":sample_bytes,"reused_frozen_vocabulary":frozen_tokenizer.is_some(),"implementation":"tokenizers 0.22.2 native Rust","original_repository_tokenizer_used":false,"special_ids":{"pad":0,"bos":1,"eos":2}},"split_rule":"SHA256(normalized text) first 8 bytes modulo 10000; test<20, validation<40, otherwise train","deduplication":"normalized exact hash and first/last 64-word fingerprint; not exhaustive contamination audit","duplicates_removed":removed,"counts":counts,"files":files,"network":"OS denied","implementation":"Rust"}),
    )?;
    Ok(())
}
pub fn benchmarks(root: &Path, out: &Path) -> Result<()> {
    let acquisition = io::json(
        &root.join("evidence/benchmark_download_manifest.json"),
        2 * 1024 * 1024,
    )?;
    for r in acquisition["files"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Missing acquisition"))?
    {
        staged(root, r)?;
    }
    ensure!(
        !out.exists() || std::fs::read_dir(out)?.next().is_none(),
        "Benchmark output must be empty"
    );
    std::fs::create_dir_all(out)?;
    let src = root.join("downloads/benchmarks");
    let mut tasks = BTreeMap::<String, Vec<Value>>::new();
    let mut boolq = Vec::new();
    parquet_rows(&src.join("boolq_validation.parquet"), |r| {
        boolq.push(json!({"id":boolq.len().to_string(),"prompt":format!("{}\nQuestion: {}?\nAnswer:",r["passage"].as_str().unwrap_or(""),r["question"].as_str().unwrap_or("").trim_end_matches('?')),"choices":[" no"," yes"],"answer":usize::from(r["answer"].as_bool().unwrap_or(false))}));
        Ok(true)
    })?;
    tasks.insert("boolq".into(), boolq);
    let mut rows = Vec::new();
    for l in BufReader::new(io::regular(
        &src.join("piqa_valid.jsonl"),
        64 * 1024 * 1024,
    )?)
    .lines()
    {
        rows.push(serde_json::from_str::<Value>(&l?)?);
    }
    let mut labels = Vec::new();
    for l in BufReader::new(io::regular(
        &src.join("piqa_valid_labels.lst"),
        1024 * 1024,
    )?)
    .lines()
    {
        labels.push(l?.parse::<usize>()?);
    }
    ensure!(rows.len() == labels.len(), "PIQA label alignment mismatch");
    let piqa=rows.iter().zip(labels).enumerate().map(|(i,(r,answer))|json!({"id":i.to_string(),"prompt":format!("Question: {}\nAnswer:",r["goal"].as_str().unwrap_or("")),"choices":[format!(" {}",r["sol1"].as_str().unwrap_or("")),format!(" {}",r["sol2"].as_str().unwrap_or(""))],"answer":answer})).collect();
    tasks.insert("piqa".into(), piqa);
    let mut arc = Vec::new();
    parquet_rows(&src.join("arc_challenge_test.parquet"), |r| {
        let labels = r["choices"]["label"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("ARC choices missing"))?;
        let answer = labels
            .iter()
            .position(|x| *x == r["answerKey"])
            .ok_or_else(|| anyhow::anyhow!("ARC answer missing"))?;
        let choices: Vec<_> = r["choices"]["text"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| format!(" {}", s.as_str().unwrap_or("")))
            .collect();
        arc.push(json!({"id":r["id"],"prompt":format!("Question: {}\nAnswer:",r["question"].as_str().unwrap_or("")),"choices":choices,"answer":answer}));
        Ok(true)
    })?;
    tasks.insert("arc_challenge".into(), arc);
    let mut files = json!({});
    for (name, rows) in tasks {
        let expected = match name.as_str() {
            "boolq" => 3270,
            "piqa" => 1838,
            _ => 1172,
        };
        ensure!(rows.len() == expected, "Unexpected benchmark size");
        let p = out.join(format!("{name}.jsonl"));
        let mut file = OpenOptions::new().write(true).create_new(true).open(&p)?;
        for r in rows {
            ensure!(
                (r["answer"].as_u64().unwrap() as usize) < r["choices"].as_array().unwrap().len(),
                "Invalid gold answer"
            );
            writeln!(file, "{r}")?;
        }
        file.sync_all()?;
        files[format!("{name}.jsonl")] =
            json!({"bytes":p.metadata()?.len(),"sha256":io::hash(&p)?,"examples":expected});
    }
    io::atomic_json(
        &out.join("manifest.json"),
        &json!({"status":"frozen","protocol":"CairnLocal-v1","files":files,"sources":acquisition,"shots":0,"network":"OS denied","notes":["Same fixed local continuation scoring; not lm-evaluation-harness-equivalent.","No complete benchmark contamination audit."]}),
    )?;
    Ok(())
}
