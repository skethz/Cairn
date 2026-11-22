use crate::io;
use anyhow::{Result, ensure};
use memmap2::Mmap;
use std::path::Path;
use tch::{Device, Tensor};
pub struct Documents {
    tokens: Mmap,
    index: Mmap,
    pub context: usize,
    pub size: u64,
    pub cursor: u64,
    a: u64,
    b: u64,
}
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
impl Documents {
    pub fn new(
        dir: &Path,
        split: &str,
        seed: u64,
        context: usize,
        limit: Option<u64>,
    ) -> Result<Self> {
        ensure!(
            ["train", "validation", "test"].contains(&split),
            "Invalid split"
        );
        ensure!((1..=4096).contains(&context), "Invalid context");
        let tf = io::regular(
            &dir.join(format!("{split}.tokens.bin")),
            32 * 1024_u64.pow(3),
        )?;
        let ix = io::regular(&dir.join(format!("{split}.index.bin")), 1024_u64.pow(3))?;
        ensure!(
            tf.metadata()?.len() % 2 == 0 && ix.metadata()?.len() % 16 == 0,
            "Invalid dataset byte counts"
        );
        let tokens = unsafe { Mmap::map(&tf)? };
        let index = unsafe { Mmap::map(&ix)? };
        let mut size = index.len() as u64 / 16;
        if let Some(n) = limit {
            size = size.min(n);
        }
        ensure!(size > 0, "Empty split");
        let mut a = 104729 + 2 * seed;
        while gcd(a, size) != 1 {
            a += 2;
        }
        let b = seed * 15485863 % size;
        Ok(Self {
            tokens,
            index,
            context,
            size,
            cursor: 0,
            a,
            b,
        })
    }
    pub fn batch(&mut self, batch: usize, device: Device) -> Result<(Tensor, Tensor)> {
        ensure!((1..=16).contains(&batch), "Invalid batch size");
        let mut ids = vec![0i64; batch * self.context];
        let mut labels = vec![-100i64; ids.len()];
        let mut longest = 0;
        for row in 0..batch {
            let pos = ((self.a as u128 * (self.cursor % self.size) as u128 + self.b as u128)
                % self.size as u128) as usize;
            self.cursor += 1;
            let offset =
                u64::from_le_bytes(self.index[pos * 16..pos * 16 + 8].try_into()?) as usize;
            let length =
                u64::from_le_bytes(self.index[pos * 16 + 8..pos * 16 + 16].try_into()?) as usize;
            ensure!(length >= 2, "Invalid document length");
            let n = (length - 1).min(self.context);
            ensure!(
                offset
                    .checked_add(n)
                    .is_some_and(|x| x < self.tokens.len() / 2),
                "Invalid document index"
            );
            longest = longest.max(n);
            for j in 0..n {
                let at = 2 * (offset + j);
                ids[row * self.context + j] =
                    u16::from_le_bytes(self.tokens[at..at + 2].try_into()?) as i64;
                labels[row * self.context + j] =
                    u16::from_le_bytes(self.tokens[at + 2..at + 4].try_into()?) as i64;
            }
        }
        let length = self.context.min(longest.div_ceil(64) * 64) as i64;
        let make = |v: &[i64]| {
            Tensor::from_slice(v)
                .reshape([batch as i64, self.context as i64])
                .narrow(1, 0, length)
                .contiguous()
                .to_device(device)
        };
        Ok((make(&ids), make(&labels)))
    }
}
pub fn verify(dir: &Path) -> Result<(serde_json::Value, String)> {
    let p = dir.join("manifest.json");
    let m = io::json(&p, 2 * 1024 * 1024)?;
    ensure!(m["status"] == "frozen", "Data is not frozen");
    for (name, r) in m["files"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Missing manifest files"))?
    {
        ensure!(
            Path::new(name).file_name().and_then(|x| x.to_str()) == Some(name.as_str()),
            "Unsafe manifest path"
        );
        let file = dir.join(name);
        ensure!(
            io::regular(&file, 32 * 1024_u64.pow(3))?.metadata()?.len()
                == r["bytes"].as_u64().unwrap_or(0),
            "Data size mismatch"
        );
        ensure!(
            io::hash(&file)? == r["sha256"].as_str().unwrap_or(""),
            "Data hash mismatch"
        );
    }
    Ok((m, io::hash(&p)?))
}
