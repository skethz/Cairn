use anyhow::{Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::Path,
};
use tch::Tensor;
pub fn regular(path: &Path, max: u64) -> Result<File> {
    ensure!(
        !path.to_string_lossy().contains("://"),
        "Only local paths are accepted"
    );
    // Reject symlink components, special files, and files replaced by a final symlink.
    let mut prefix = std::path::PathBuf::new();
    for part in path.components() {
        prefix.push(part);
        ensure!(
            !std::fs::symlink_metadata(&prefix)?.file_type().is_symlink(),
            "Symlink path rejected: {}",
            prefix.display()
        );
    }
    use std::os::unix::fs::OpenOptionsExt;
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let m = f.metadata()?;
    ensure!(
        m.is_file() && m.len() <= max,
        "Invalid or oversized local file: {}",
        path.display()
    );
    Ok(f)
}
pub fn json(path: &Path, max: u64) -> Result<Value> {
    let mut s = String::new();
    regular(path, max)?.read_to_string(&mut s)?;
    Ok(serde_json::from_str(&s)?)
}
pub fn hash(path: &Path) -> Result<String> {
    let mut f = regular(path, 64 * 1024_u64.pow(3))?;
    let mut h = Sha256::new();
    let mut b = vec![0; 8 * 1024 * 1024];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(hex::encode(h.finalize()))
}
pub fn atomic_json(path: &Path, value: &Value) -> Result<()> {
    let temp = path.with_extension("partial");
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    f.write_all(&serde_json::to_vec_pretty(value)?)?;
    f.sync_all()?;
    std::fs::rename(temp, path)?;
    Ok(())
}
pub fn tensors(path: &Path, max: u64) -> Result<Vec<(String, Tensor)>> {
    // Native parser never accepts pickle, TorchScript, or arbitrary object files.
    let mut file = regular(path, max)?;
    let len = file.metadata()?.len();
    ensure!(len >= 8, "Truncated tensor file");
    let mut first = [0u8; 8];
    file.read_exact(&mut first)?;
    let header = u64::from_le_bytes(first);
    ensure!(
        header > 0 && header <= 64 * 1024 * 1024 && header + 8 <= len,
        "Unbounded tensor header"
    );
    ensure!(
        path.extension().and_then(|x| x.to_str()) == Some("safetensors"),
        "Only safetensors are accepted"
    );
    let mapped = unsafe { memmap2::Mmap::map(&file)? };
    let metadata: Value = serde_json::from_slice(&mapped[8..8 + header as usize])?;
    let base = 8 + header as usize;
    let mut layouts = Vec::new();
    for (name, entry) in metadata
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid tensor header"))?
    {
        if name == "__metadata__" {
            continue;
        }
        let kind = match entry["dtype"].as_str().unwrap_or("") {
            "F64" => tch::Kind::Double,
            "F32" => tch::Kind::Float,
            "BF16" => tch::Kind::BFloat16,
            "F16" => tch::Kind::Half,
            "F8_E4M3" => tch::Kind::Float8e4m3fn,
            "F8_E5M2" => tch::Kind::Float8e5m2,
            "I64" => tch::Kind::Int64,
            "I32" => tch::Kind::Int,
            "I16" => tch::Kind::Int16,
            "I8" => tch::Kind::Int8,
            "U8" => tch::Kind::Uint8,
            "BOOL" => tch::Kind::Bool,
            _ => anyhow::bail!("Unsupported tensor dtype"),
        };
        let shape: Vec<i64> = serde_json::from_value(entry["shape"].clone())?;
        ensure!(
            shape.len() <= 16 && shape.iter().all(|d| *d >= 0),
            "Invalid tensor shape"
        );
        let count = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d as usize))
            .ok_or_else(|| anyhow::anyhow!("Tensor shape overflow"))?;
        let bytes = count
            .checked_mul(kind.elt_size_in_bytes())
            .ok_or_else(|| anyhow::anyhow!("Tensor byte overflow"))?;
        let offsets: Vec<usize> = serde_json::from_value(entry["data_offsets"].clone())?;
        ensure!(
            offsets.len() == 2
                && offsets[1] >= offsets[0]
                && offsets[1] - offsets[0] == bytes
                && offsets[1] <= mapped.len() - base,
            "Invalid tensor offsets"
        );
        layouts.push((offsets[0], offsets[1], name.clone(), shape, kind));
    }
    layouts.sort_by_key(|x| (x.0, x.1));
    let mut end = 0;
    for l in &layouts {
        ensure!(l.0 == end, "Overlapping or noncontiguous tensors");
        end = l.1;
    }
    ensure!(end == mapped.len() - base, "Trailing tensor bytes");
    layouts
        .into_iter()
        .map(|(start, end, name, shape, kind)| {
            Ok((
                name,
                Tensor::f_from_data_size(&mapped[base + start..base + end], &shape, kind)?,
            ))
        })
        .collect()
}
