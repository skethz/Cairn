use crate::io;
use anyhow::{Result, ensure};
use std::{
    collections::{HashMap, HashSet},
    io::Read,
    path::Path,
};
pub fn load(path: &Path) -> Result<tokenizers::Tokenizer> {
    let mut bytes = Vec::new();
    io::regular(path, 16 * 1024 * 1024)?.read_to_end(&mut bytes)?;
    tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("Tokenizer JSON: {e}"))
}
pub fn encode(t: &tokenizers::Tokenizer, text: &str) -> Result<Vec<i64>> {
    ensure!(text.len() <= 1_000_000, "Text exceeds budget");
    Ok(t.encode(text, false)
        .map_err(|e| anyhow::anyhow!("Tokenization: {e}"))?
        .get_ids()
        .iter()
        .map(|x| *x as i64)
        .collect())
}
/// Historical local inference vocabulary, distinct from the frozen training BPE.
pub struct ByteBpe {
    pub merges: Vec<(u32, u32)>,
    pieces: Vec<Vec<u8>>,
}
impl ByteBpe {
    pub fn new(merges: Vec<(u32, u32)>) -> Result<Self> {
        ensure!(merges.len() <= 131072 - 259, "Too many merges");
        let mut pieces = vec![b"<pad>".to_vec(), b"<bos>".to_vec(), b"<eos>".to_vec()];
        pieces.extend((0..256).map(|b| vec![b as u8]));
        let mut seen = HashSet::new();
        let mut size = 256;
        for &(l, r) in &merges {
            ensure!(
                l >= 3
                    && r >= 3
                    && (l as usize) < pieces.len()
                    && (r as usize) < pieces.len()
                    && seen.insert((l, r)),
                "Invalid merge"
            );
            let p = [pieces[l as usize].as_slice(), pieces[r as usize].as_slice()].concat();
            size += p.len();
            ensure!(
                p.len() <= 65536 && size <= 64 * 1024 * 1024,
                "Vocabulary budget exceeded"
            );
            pieces.push(p);
        }
        Ok(Self { merges, pieces })
    }
    pub fn load(path: &Path) -> Result<Self> {
        let v = io::json(path, 8_000_000)?;
        ensure!(
            v.as_object().is_some_and(|x| x.len() == 3)
                && v["format"] == "numbat-byte-bpe"
                && v["version"] == 1,
            "Unsupported byte BPE"
        );
        Self::new(serde_json::from_value(v["merges"].clone())?)
    }
    fn replace(seq: &[u32], pair: (u32, u32), id: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < seq.len() {
            if i + 1 < seq.len() && (seq[i], seq[i + 1]) == pair {
                out.push(id);
                i += 2;
            } else {
                out.push(seq[i]);
                i += 1;
            }
        }
        out
    }
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<u32>> {
        ensure!(
            text.len() <= 1_000_000 && text.len() * self.merges.len().max(1) <= 20_000_000,
            "Tokenization budget exceeded"
        );
        let mut seq: Vec<_> = text.bytes().map(|x| x as u32 + 3).collect();
        for (i, &pair) in self.merges.iter().enumerate() {
            seq = Self::replace(&seq, pair, 259 + i as u32);
        }
        if bos {
            seq.insert(0, 1);
        }
        if eos {
            seq.push(2);
        }
        Ok(seq)
    }
    pub fn decode(&self, ids: &[u32], skip_special: bool, errors: &str) -> Result<String> {
        ensure!(ids.len() <= 1_000_000, "Too many tokens");
        let mut out = Vec::new();
        for &id in ids {
            ensure!((id as usize) < self.pieces.len(), "Invalid token");
            if id < 3 && skip_special {
                continue;
            }
            out.extend(&self.pieces[id as usize]);
            ensure!(out.len() <= 1_000_000, "Decoded budget exceeded");
        }
        match errors {
            "strict" => Ok(String::from_utf8(out)?),
            "replace" => Ok(String::from_utf8_lossy(&out).into_owned()),
            "ignore" => {
                let mut text = String::new();
                let mut remaining = out.as_slice();
                while !remaining.is_empty() {
                    match std::str::from_utf8(remaining) {
                        Ok(s) => {
                            text.push_str(s);
                            break;
                        }
                        Err(e) => {
                            text.push_str(std::str::from_utf8(&remaining[..e.valid_up_to()])?);
                            let n = e.valid_up_to()
                                + e.error_len().unwrap_or(remaining.len() - e.valid_up_to());
                            remaining = &remaining[n..];
                        }
                    }
                }
                Ok(text)
            }
            _ => anyhow::bail!("Unknown decoding policy"),
        }
    }
    pub fn train(texts: &[String], size: usize, minimum: usize) -> Result<Self> {
        ensure!(
            (259..=131072).contains(&size) && minimum > 0 && texts.len() <= 10000,
            "Invalid training budget"
        );
        let total: usize = texts.iter().map(|x| x.len()).sum();
        ensure!(
            total <= 1_000_000 && total * (size - 259).max(1) <= 20_000_000,
            "Training budget exceeded"
        );
        let mut docs: Vec<Vec<u32>> = texts
            .iter()
            .map(|s| s.bytes().map(|b| b as u32 + 3).collect())
            .collect();
        let mut merges = Vec::new();
        while merges.len() + 259 < size {
            let mut counts = HashMap::new();
            for seq in &docs {
                for p in seq.windows(2) {
                    *counts.entry((p[0], p[1])).or_insert(0usize) += 1;
                }
            }
            let Some((&pair, &count)) = counts
                .iter()
                .min_by_key(|(p, n)| (std::cmp::Reverse(**n), **p))
            else {
                break;
            };
            if count < minimum {
                break;
            }
            for seq in &mut docs {
                *seq = Self::replace(seq, pair, 259 + merges.len() as u32);
            }
            merges.push(pair);
        }
        Self::new(merges)
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        ensure!(!path.exists(), "Refusing to overwrite vocabulary");
        io::atomic_json(
            path,
            &serde_json::json!({"format":"numbat-byte-bpe","version":1,"merges":self.merges}),
        )
    }
}
