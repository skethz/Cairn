//! Owned training architecture. Names and operator ordering follow the frozen Python oracle.
use crate::{native, recurrence::chunk_delta};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tch::{Device, Kind, Reduction, Tensor};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub family: String,
    pub width: i64,
    pub layers: i64,
    pub vocab_size: i64,
    pub gdn_heads: i64,
    pub gdn_key_dim: i64,
    pub gdn_value_dim: i64,
    pub gdn_chunk: i64,
    pub kda_chunk: i64,
    pub mla_heads: i64,
    pub mla_rank: i64,
    pub mla_content_dim: i64,
    pub mla_rope_dim: i64,
    pub mla_value_dim: i64,
    pub gqa_kv_heads: i64,
    pub experts: i64,
    pub top_k: i64,
    pub groups: i64,
    pub keep_groups: i64,
    pub expert_width: i64,
    pub dense_width: i64,
    #[serde(default)]
    pub dense_layer_indices: Option<Vec<i64>>,
    #[serde(default = "yes")]
    pub activation_checkpointing: bool,
}
fn yes() -> bool {
    true
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            [
                "cairn",
                "gdn_h_recipe_proxy",
                "kda_mla_operator",
                "gdn_gqa",
                "full_mla",
                "dense_gqa"
            ]
            .contains(&self.family.as_str()),
            "Unknown model family"
        );
        for (name, x, lo, hi) in [
            ("width", self.width, 8, 2048),
            ("layers", self.layers, 1, 32),
            ("vocabulary", self.vocab_size, 4, 65536),
            ("experts", self.experts, 2, 64),
            ("expert_width", self.expert_width, 4, 4096),
            ("dense_width", self.dense_width, 4, 8192),
            ("heads", self.gdn_heads, 1, 32),
            ("key", self.gdn_key_dim, 2, 128),
            ("value", self.gdn_value_dim, 2, 128),
            ("mla_heads", self.mla_heads, 1, 32),
            ("rank", self.mla_rank, 2, 1024),
            ("content", self.mla_content_dim, 2, 128),
            ("rope", self.mla_rope_dim, 2, 128),
            ("mla_value", self.mla_value_dim, 2, 128),
            ("gdn_chunk", self.gdn_chunk, 1, 128),
            ("kda_chunk", self.kda_chunk, 1, 128),
        ] {
            ensure!((lo..=hi).contains(&x), "Out of bounds {name}");
        }
        ensure!(
            self.groups > 0 && self.experts % self.groups == 0 && self.experts / self.groups >= 2,
            "Invalid expert groups"
        );
        ensure!(
            self.keep_groups > 0
                && self.keep_groups <= self.groups
                && self.top_k > 0
                && self.top_k <= self.keep_groups * (self.experts / self.groups),
            "Invalid routing capacity"
        );
        ensure!(
            self.gqa_kv_heads > 0
                && self.mla_heads % self.gqa_kv_heads == 0
                && self.mla_rope_dim % 2 == 0
                && self.mla_value_dim % 2 == 0,
            "Invalid attention dimensions"
        );
        if let Some(indices) = &self.dense_layer_indices {
            ensure!(
                indices.iter().all(|i| *i >= 0 && *i < self.layers),
                "Invalid dense layer index"
            );
        }
        Ok(())
    }
    pub fn global(&self, i: i64) -> bool {
        ["full_mla", "dense_gqa"].contains(&self.family.as_str()) || i % 4 == 3
    }
    pub fn sparse(&self, i: i64) -> bool {
        self.family != "dense_gqa"
            && !self
                .dense_layer_indices
                .clone()
                .unwrap_or_else(|| vec![0, self.layers - 1])
                .contains(&i)
    }
}
pub struct Parameter {
    pub name: String,
    pub tensor: Tensor,
}
pub struct Model {
    pub config: Config,
    pub params: Vec<Parameter>,
    lookup: HashMap<String, usize>,
}
#[derive(Clone)]
struct Spec {
    name: String,
    shape: Vec<i64>,
    value: f64,
    random: bool,
}
fn specs(c: &Config) -> Vec<Spec> {
    let mut s = Vec::new();
    fn add(s: &mut Vec<Spec>, n: String, shape: Vec<i64>, value: f64, random: bool) {
        s.push(Spec {
            name: n,
            shape,
            value,
            random,
        });
    }
    fn weight(s: &mut Vec<Spec>, n: &str, shape: Vec<i64>) {
        add(s, format!("{n}.weight"), shape, 0., true);
    }
    fn norm(s: &mut Vec<Spec>, n: &str, d: i64) {
        add(s, format!("{n}.weight"), vec![d], 1., false);
    }
    fn mlp(s: &mut Vec<Spec>, n: &str, d: i64, m: i64) {
        weight(s, &format!("{n}.gate"), vec![m, d]);
        weight(s, &format!("{n}.up"), vec![m, d]);
        weight(s, &format!("{n}.down"), vec![d, m]);
    }
    let d = c.width;
    weight(&mut s, "embedding", vec![c.vocab_size, d]);
    for i in 0..c.layers {
        let p = format!("blocks.{i}");
        let m = format!("{p}.mixer");
        let f = format!("{p}.ffn");
        norm(&mut s, &format!("{p}.norm1"), d);
        norm(&mut s, &format!("{p}.norm2"), d);
        if !c.global(i) {
            let (hk, hv) = (c.gdn_heads * c.gdn_key_dim, c.gdn_heads * c.gdn_value_dim);
            let channels = if c.family == "kda_mla_operator" {
                c.gdn_key_dim
            } else {
                1
            };
            add(
                &mut s,
                format!("{m}.A_log"),
                vec![c.gdn_heads, channels],
                0.,
                false,
            );
            add(
                &mut s,
                format!("{m}.dt_bias"),
                vec![c.gdn_heads, channels],
                0.03_f64.exp_m1().ln(),
                false,
            );
            for (n, o, inp) in [
                ("q", hk, d),
                ("k", hk, d),
                ("v", hv, d),
                ("gate", hv, d),
                ("out", d, hv),
            ] {
                weight(&mut s, &format!("{m}.{n}"), vec![o, inp]);
            }
            for (n, w) in [("qconv", hk), ("kconv", hk), ("vconv", hv)] {
                weight(&mut s, &format!("{m}.{n}"), vec![w, 1, 4]);
            }
            weight(&mut s, &format!("{m}.a"), vec![c.gdn_heads * channels, d]);
            weight(&mut s, &format!("{m}.b"), vec![c.gdn_heads, d]);
            norm(&mut s, &format!("{m}.norm"), c.gdn_value_dim);
        } else if ["gdn_gqa", "dense_gqa"].contains(&c.family.as_str()) {
            for (n, h) in [
                ("q", c.mla_heads),
                ("k", c.gqa_kv_heads),
                ("v", c.gqa_kv_heads),
            ] {
                weight(&mut s, &format!("{m}.{n}"), vec![h * c.mla_value_dim, d]);
            }
            if c.family != "dense_gqa" {
                weight(
                    &mut s,
                    &format!("{m}.gate"),
                    vec![c.mla_heads * c.mla_value_dim, d],
                );
            }
            weight(
                &mut s,
                &format!("{m}.out"),
                vec![d, c.mla_heads * c.mla_value_dim],
            );
        } else {
            weight(&mut s, &format!("{m}.qa"), vec![c.mla_rank, d]);
            norm(&mut s, &format!("{m}.qnorm"), c.mla_rank);
            weight(
                &mut s,
                &format!("{m}.qb"),
                vec![
                    c.mla_heads * (c.mla_content_dim + c.mla_rope_dim),
                    c.mla_rank,
                ],
            );
            weight(
                &mut s,
                &format!("{m}.kva"),
                vec![c.mla_rank + c.mla_rope_dim, d],
            );
            norm(&mut s, &format!("{m}.kvnorm"), c.mla_rank);
            weight(
                &mut s,
                &format!("{m}.kvb"),
                vec![
                    c.mla_heads * (c.mla_content_dim + c.mla_value_dim),
                    c.mla_rank,
                ],
            );
            if c.family != "gdn_h_recipe_proxy" {
                weight(
                    &mut s,
                    &format!("{m}.gate"),
                    vec![c.mla_heads * c.mla_value_dim, d],
                );
            }
            weight(
                &mut s,
                &format!("{m}.out"),
                vec![d, c.mla_heads * c.mla_value_dim],
            );
        }
        if c.sparse(i) {
            weight(&mut s, &format!("{f}.router"), vec![c.experts, d]);
            for e in 0..c.experts {
                mlp(&mut s, &format!("{f}.experts.{e}"), d, c.expert_width);
            }
            mlp(&mut s, &format!("{f}.shared"), d, c.expert_width);
        } else {
            mlp(&mut s, &f, d, c.dense_width);
        }
    }
    norm(&mut s, "norm", d);
    weight(&mut s, "head", vec![c.vocab_size, d]);
    s
}
impl Model {
    pub fn parameter_count(c: &Config) -> Result<u64> {
        c.validate()?;
        Ok(specs(c)
            .iter()
            .map(|s| s.shape.iter().product::<i64>() as u64)
            .sum())
    }
    pub fn new(c: Config, seed: u64, device: Device) -> Result<Self> {
        ensure!(
            Self::parameter_count(&c)? <= 2_000_000_000,
            "Model exceeds allocation ceiling"
        );
        let mut params = Vec::new();
        let mut lookup = HashMap::new();
        let layout = specs(&c);
        for s in &layout {
            let mut t = Tensor::full(s.shape.as_slice(), s.value, (Kind::Float, Device::Cpu));
            // Preserve global CPU RNG consumption of nn module construction.
            // These values are overwritten below, but RNG checkpoints are observable.
            if s.random {
                if s.name == "embedding.weight" {
                    let _ = t.normal_(0., 1.);
                } else {
                    let _ = t.uniform_(-1., 1.);
                }
            }
            lookup.insert(s.name.clone(), params.len());
            params.push(Parameter {
                name: s.name.clone(),
                tensor: t,
            });
        }
        for (s, p) in layout.iter().zip(&mut params) {
            if s.random {
                let _ = p.tensor.normal_(0., 0.02);
            }
        }
        for (s, p) in layout.iter().zip(&mut params) {
            let module = s.name.strip_suffix(".weight").unwrap_or("");
            if s.random && (module.ends_with(".out") || module.ends_with(".down")) {
                let _ = p.tensor.normal_(0., 0.02 / ((2 * c.layers) as f64).sqrt());
            }
        }
        for (s, p) in layout.iter().zip(&mut params) {
            if s.random {
                let module = s.name.strip_suffix(".weight").unwrap();
                let hash = Sha256::digest(format!("{seed}:{module}"));
                let number = u64::from_le_bytes(hash[..8].try_into().unwrap());
                let std = if module.ends_with(".out") || module.ends_with(".down") {
                    0.02 / ((2 * c.layers) as f64).sqrt()
                } else {
                    0.02
                };
                native::normal(&p.tensor, number, std)?;
            }
            p.tensor = p.tensor.to_device(device).set_requires_grad(true);
        }
        Ok(Self {
            config: c,
            params,
            lookup,
        })
    }
    pub fn tensor(&self, n: &str) -> &Tensor {
        &self.params[*self
            .lookup
            .get(n)
            .unwrap_or_else(|| panic!("Missing parameter {n}"))]
        .tensor
    }
    pub fn linear(&self, n: &str, x: &Tensor) -> Tensor {
        x.linear(self.tensor(&format!("{n}.weight")), None::<&Tensor>)
    }
    pub fn norm(&self, n: &str, x: &Tensor) -> Tensor {
        rms(x, self.tensor(&format!("{n}.weight")), 1e-6)
    }
    pub fn embed(&self, ids: &Tensor) -> Tensor {
        Tensor::embedding(self.tensor("embedding.weight"), ids, -1, false, false)
    }
    pub fn mlp(&self, n: &str, x: &Tensor) -> Tensor {
        self.linear(
            &format!("{n}.down"),
            &(self.linear(&format!("{n}.gate"), x).silu() * self.linear(&format!("{n}.up"), x)),
        )
    }
    fn conv(&self, n: &str, x: Tensor) -> Tensor {
        let w = self.tensor(&format!("{n}.weight"));
        x.transpose(1, 2)
            .constant_pad_nd([3, 0])
            .conv1d(w, None::<&Tensor>, [1], [0], [1], w.size()[0])
            .transpose(1, 2)
            .silu()
    }
    fn delta(&self, n: &str, x: &Tensor) -> Tensor {
        let c = &self.config;
        let (b, t, h, dk, dv) = (
            x.size()[0],
            x.size()[1],
            c.gdn_heads,
            c.gdn_key_dim,
            c.gdn_value_dim,
        );
        let q = normalize(
            &self
                .conv(&format!("{n}.qconv"), self.linear(&format!("{n}.q"), x))
                .reshape([b, t, h, dk]),
        );
        let k = normalize(
            &self
                .conv(&format!("{n}.kconv"), self.linear(&format!("{n}.k"), x))
                .reshape([b, t, h, dk]),
        );
        let v = self
            .conv(&format!("{n}.vconv"), self.linear(&format!("{n}.v"), x))
            .reshape([b, t, h, dv]);
        let decay = -self.tensor(&format!("{n}.A_log")).exp()
            * (self
                .linear(&format!("{n}.a"), x)
                .to_kind(Kind::Float)
                .reshape([b, t, h, -1])
                + self.tensor(&format!("{n}.dt_bias")))
            .softplus();
        let beta = self
            .linear(&format!("{n}.b"), x)
            .to_kind(Kind::Float)
            .sigmoid()
            .transpose(1, 2);
        let chunk = if c.family == "kda_mla_operator" {
            c.kda_chunk
        } else {
            c.gdn_chunk
        };
        let (out, _) = chunk_delta(
            &q.transpose(1, 2),
            &k.transpose(1, 2),
            &v.transpose(1, 2),
            &decay.permute([0, 2, 1, 3]),
            &beta,
            chunk,
            None,
        );
        let out = self
            .norm(&format!("{n}.norm"), &out)
            .transpose(1, 2)
            .reshape([b, t, -1])
            .to_kind(x.kind());
        let gate = self.linear(&format!("{n}.gate"), x);
        let gate = if c.family == "gdn_h_recipe_proxy" {
            gate.sigmoid()
        } else {
            gate.silu()
        };
        self.linear(&format!("{n}.out"), &(out * gate))
    }
    fn latent(&self, n: &str, x: &Tensor) -> Tensor {
        let c = &self.config;
        let (b, t, h) = (x.size()[0], x.size()[1], c.mla_heads);
        let q = self
            .linear(
                &format!("{n}.qb"),
                &self.norm(&format!("{n}.qnorm"), &self.linear(&format!("{n}.qa"), x)),
            )
            .reshape([b, t, h, -1])
            .transpose(1, 2);
        let latent = self.linear(&format!("{n}.kva"), x);
        let kv = self
            .linear(
                &format!("{n}.kvb"),
                &self.norm(&format!("{n}.kvnorm"), &latent.narrow(-1, 0, c.mla_rank)),
            )
            .reshape([b, t, h, -1])
            .transpose(1, 2);
        let mut kr = latent
            .narrow(-1, c.mla_rank, c.mla_rope_dim)
            .unsqueeze(1)
            .expand([-1, h, -1, -1], false);
        let mut qr = q.narrow(-1, c.mla_content_dim, c.mla_rope_dim);
        if c.family != "gdn_h_recipe_proxy" {
            let positions = Tensor::arange(t, (Kind::Int64, x.device())).expand([b, t], false);
            qr = rotary(&qr, &positions, 1e6);
            kr = rotary(&kr, &positions, 1e6);
        }
        let q = Tensor::cat(&[q.narrow(-1, 0, c.mla_content_dim), qr], -1);
        let k = Tensor::cat(&[kv.narrow(-1, 0, c.mla_content_dim), kr], -1);
        let v = kv.narrow(-1, c.mla_content_dim, c.mla_value_dim);
        let mut out = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.,
            true,
            None,
            false,
        )
        .transpose(1, 2)
        .reshape([b, t, -1]);
        if c.family != "gdn_h_recipe_proxy" {
            out = out * self.linear(&format!("{n}.gate"), x).sigmoid();
        }
        self.linear(&format!("{n}.out"), &out)
    }
    fn grouped(&self, n: &str, x: &Tensor) -> Tensor {
        let c = &self.config;
        let (b, t, h, kh, d) = (
            x.size()[0],
            x.size()[1],
            c.mla_heads,
            c.gqa_kv_heads,
            c.mla_value_dim,
        );
        let positions = Tensor::arange(t, (Kind::Int64, x.device())).expand([b, t], false);
        let q = rotary(
            &self
                .linear(&format!("{n}.q"), x)
                .reshape([b, t, h, d])
                .transpose(1, 2),
            &positions,
            1e6,
        );
        let k = rotary(
            &self
                .linear(&format!("{n}.k"), x)
                .reshape([b, t, kh, d])
                .transpose(1, 2),
            &positions,
            1e6,
        )
        .repeat_interleave_self_int(h / kh, Some(1), None);
        let v = self
            .linear(&format!("{n}.v"), x)
            .reshape([b, t, kh, d])
            .transpose(1, 2)
            .repeat_interleave_self_int(h / kh, Some(1), None);
        let mut out = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.,
            true,
            None,
            false,
        )
        .transpose(1, 2)
        .reshape([b, t, -1]);
        if c.family != "dense_gqa" {
            out = out * self.linear(&format!("{n}.gate"), x).sigmoid();
        }
        self.linear(&format!("{n}.out"), &out)
    }
    fn sparse(&self, n: &str, x: &Tensor, valid: &Tensor) -> (Tensor, Tensor, Tensor) {
        let c = &self.config;
        let flat = x.reshape([-1, c.width]);
        let scores = flat
            .to_kind(Kind::Float)
            .linear(
                &self
                    .tensor(&format!("{n}.router.weight"))
                    .to_kind(Kind::Float),
                None::<&Tensor>,
            )
            .sigmoid();
        let grouped = scores
            .reshape([-1, c.groups, c.experts / c.groups])
            .topk(2, -1, true, true)
            .0
            .sum_dim_intlist([-1].as_slice(), false, None);
        let retained = grouped.topk(c.keep_groups, -1, true, true).1;
        let membership = Tensor::arange(c.experts, (Kind::Int64, x.device()))
            .floor_divide_scalar(c.experts / c.groups);
        let candidates = retained
            .unsqueeze(-1)
            .eq_tensor(&membership)
            .any_dim(-2, false);
        let selected = scores
            .masked_fill(&candidates.logical_not(), f64::NEG_INFINITY)
            .topk(c.top_k, -1, true, true)
            .1;
        let weights = scores.gather(-1, &selected, false);
        let weights =
            (&weights / weights.sum_dim_intlist([-1].as_slice(), true, None)).to_kind(x.kind());
        let mut output = Tensor::zeros_like(&flat);
        let dispatch = if native::optimized() && x.device().is_cuda() {
            Some(native::routes(&selected, c.experts).expect("Bounded native route dispatch"))
        } else { None };
        for e in 0..c.experts {
            let (rows, columns) = if let Some((routes, counts)) = &dispatch {
                let indices=routes.select(0,e).narrow(0,0,counts[e as usize]);
                (indices.floor_divide_scalar(c.top_k), indices.remainder(c.top_k))
            } else {
                let indices = selected.eq(e).nonzero();
                (indices.select(1, 0), indices.select(1, 1))
            };
            let w = weights.index(&[Some(&rows), Some(&columns)]).unsqueeze(-1);
            let contribution =
                self.mlp(&format!("{n}.experts.{e}"), &flat.index_select(0, &rows)) * w;
            if native::optimized() {
                let _ = output.index_add_(0, &rows, &contribution);
            } else {
                output = output.index_add(0, &rows, &contribution);
            }
        }
        let mask = valid.reshape([-1]).to_kind(Kind::Float);
        let count = mask.sum(None).clamp_min(1.);
        let normalized = &scores / scores.sum_dim_intlist([-1].as_slice(), true, None);
        let pbar =
            (normalized * mask.unsqueeze(1)).sum_dim_intlist([0].as_slice(), false, None) / &count;
        let load = Tensor::zeros([c.experts], (Kind::Float, x.device())).scatter_add(
            0,
            &selected.reshape([-1]),
            &mask.unsqueeze(1).expand_as(&selected).reshape([-1]),
        );
        let frequency = load.detach() / (&count * c.top_k);
        let aux = (&frequency * pbar).sum(None) * c.experts;
        (
            self.mlp(&format!("{n}.shared"), x) + output.reshape_as(x),
            aux,
            frequency,
        )
    }
    pub fn block(&self, i: i64, x: &Tensor, valid: &Tensor) -> (Tensor, Tensor, Tensor) {
        let p = format!("blocks.{i}");
        let n = format!("{p}.mixer");
        let z = self.norm(&format!("{p}.norm1"), x);
        let mix = if !self.config.global(i) {
            self.delta(&n, &z)
        } else if ["gdn_gqa", "dense_gqa"].contains(&self.config.family.as_str()) {
            self.grouped(&n, &z)
        } else {
            self.latent(&n, &z)
        };
        let x = x + mix;
        let z = self.norm(&format!("{p}.norm2"), &x);
        let (y, aux, load) = if self.config.sparse(i) {
            self.sparse(&format!("{p}.ffn"), &z, valid)
        } else {
            (
                self.mlp(&format!("{p}.ffn"), &z),
                Tensor::zeros([], (Kind::Float, x.device())),
                Tensor::zeros([1], (Kind::Float, x.device())),
            )
        };
        (x + y, aux, load)
    }
    pub fn hidden(&self, ids: &Tensor, valid: &Tensor) -> (Tensor, Tensor, Tensor) {
        let mut x = self.embed(ids);
        let mut aux = Vec::new();
        let mut loads = Vec::new();
        for i in 0..self.config.layers {
            let (y, a, l) = self.block(i, &x, valid);
            x = y;
            if self.config.sparse(i) {
                aux.push(a);
                loads.push(l.detach());
            }
        }
        let a = if aux.is_empty() {
            Tensor::zeros([], (Kind::Float, x.device()))
        } else {
            Tensor::stack(&aux, 0).mean(None)
        };
        let l = if loads.is_empty() {
            Tensor::zeros([1], (Kind::Float, x.device()))
        } else {
            Tensor::stack(&loads, 0).mean_dim([0].as_slice(), false, None)
        };
        (x, a, l)
    }
    pub fn logits(&self, ids: &Tensor, valid: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (x, a, l) = self.hidden(ids, valid);
        (self.linear("head", &self.norm("norm", &x)), a, l)
    }
    pub fn loss(&self, ids: &Tensor, labels: &Tensor) -> (Tensor, Tensor, Tensor) {
        let (x, a, l) = self.hidden(ids, &labels.ne(-100));
        let x = self.norm("norm", &x);
        let mut total = Tensor::zeros([], (Kind::Float, x.device()));
        for start in (0..x.size()[1]).step_by(256) {
            let n = 256.min(x.size()[1] - start);
            total = total
                + ce(
                    &self.linear("head", &x.narrow(1, start, n)),
                    &labels.narrow(1, start, n),
                );
        }
        (total, a, l)
    }
    pub fn zero_grad(&self) -> Result<()> {
        for p in &self.params {
            native::clear_grad(&p.tensor)?;
        }
        Ok(())
    }
    pub fn load(&mut self, ts: Vec<(String, Tensor)>) -> Result<()> {
        ensure!(
            ts.len() == self.params.len(),
            "Checkpoint parameter count mismatch"
        );
        let mut seen = std::collections::HashSet::new();
        tch::no_grad(|| -> Result<()> {
            for (name, t) in ts {
                ensure!(seen.insert(name.clone()), "Duplicate tensor");
                let i = *self
                    .lookup
                    .get(&name)
                    .ok_or_else(|| anyhow::anyhow!("Unexpected parameter {name}"))?;
                let p = &mut self.params[i].tensor;
                ensure!(
                    t.size() == p.size() && t.kind() == p.kind(),
                    "Shape or dtype mismatch: {name}"
                );
                p.copy_(&t.to_device(p.device()));
            }
            Ok(())
        })
    }
    /// Recompute one block at a time. No dropout or stochastic model operations.
    pub fn backward_recompute(
        &self,
        ids: &Tensor,
        labels: &Tensor,
        valid_total: i64,
        router_coefficient: f64,
        bf16: bool,
    ) -> Result<(f64, f64, Tensor)> {
        let count = labels.ne(-100).sum(Kind::Int64).int64_value(&[]);
        let scale = Tensor::full(
            [],
            router_coefficient * count as f64 / valid_total as f64,
            (Kind::Float, ids.device()),
        );
        self.backward_recompute_scaled(ids, labels, valid_total, &scale, bf16)
    }
    pub fn backward_recompute_scaled(
        &self,
        ids: &Tensor,
        labels: &Tensor,
        valid_total: i64,
        router_scale: &Tensor,
        bf16: bool,
    ) -> Result<(f64, f64, Tensor)> {
        ensure!(valid_total > 0, "Backward requires scored target tokens");
        let valid = labels.ne(-100);
        let mut states = Vec::new();
        let mut retained = Vec::new();
        let mut auxes = Vec::new();
        let mut loads = Vec::new();
        let last = tch::no_grad(|| {
            crate::native::autocast(bf16, || {
                let mut x = self.embed(ids);
                for i in 0..self.config.layers {
                    let keep = (i as usize) < native::retained();
                    let input = x.detach().set_requires_grad(keep);
                    let (y, a, l) = if keep {
                        tch::with_grad(|| self.block(i, &input, &valid))
                    } else { self.block(i, &input, &valid) };
                    retained.push(if keep { Some((y.shallow_clone(), a.shallow_clone())) } else { None });
                    states.push(input);
                    x = y.detach();
                    if self.config.sparse(i) {
                        auxes.push(a.detach());
                        loads.push(l.detach());
                    }
                }
                x
            })
        });
        let ns = auxes.len();
        let aux = if ns == 0 {
            0.
        } else {
            Tensor::stack(&auxes, 0).mean(None).double_value(&[])
        };
        let load = if ns == 0 {
            Tensor::zeros([1], (Kind::Float, ids.device()))
        } else {
            Tensor::stack(&loads, 0).mean_dim([0].as_slice(), false, None)
        };
        let last = last.detach().set_requires_grad(true);
        // Normalize the complete hidden sequence once, as in Python. Reducing
        // its parameter gradient independently by tile changes FP32 sum order.
        let normalized = self.norm("norm", &last);
        let leaf = normalized.detach().set_requires_grad(true);
        // Python autocast reuses a single BF16 head-weight cast across every
        // loss tile. Accumulate tile gradients in BF16, then cast ONCE to the
        // FP32 master parameter. Independent casts per tile change rounding.
        let cast_head = if bf16 { Some(self.tensor("head.weight").to_kind(Kind::BFloat16)) } else { None };
        let head_leaf = cast_head.as_ref().map(|w| w.detach().set_requires_grad(true));
        let mut losses = Vec::new();
        let starts: Vec<_> = (0..last.size()[1]).step_by(256).collect();
        // The Python sum graph visits the newest loss tile first in backward.
        for start in starts.into_iter().rev() {
            let n = 256.min(last.size()[1] - start);
            let value = crate::native::autocast(bf16, || {
                ce(
                    &if let Some(w) = &head_leaf {
                        leaf.narrow(1, start, n).linear(w, None::<&Tensor>)
                    } else { self.linear("head", &leaf.narrow(1, start, n)) },
                    &labels.narrow(1, start, n),
                )
            });
            losses.push(value.detach());
            (value / valid_total).backward();
        }
        if let (Some(cast), Some(leaf)) = (&cast_head, &head_leaf) {
            native::backward(cast, &leaf.grad())?;
        }
        native::backward(&normalized, &leaf.grad())?;
        let mut total = Tensor::zeros([], (Kind::Float, ids.device()));
        for loss in losses.into_iter().rev() {
            total = total + loss;
        }
        let loss = total.double_value(&[]);
        let mut grad = last.grad().detach();
        for i in (0..self.config.layers).rev() {
            let input = states.pop().unwrap().set_requires_grad(true);
            let (out, a) = if let Some(pair) = retained.pop().unwrap() { pair } else {
                let (out, a, _) = crate::native::autocast(bf16, || self.block(i, &input, &valid));
                (out, a)
            };
            if self.config.sparse(i) {
                let objective = (&out * &grad).sum(None) + a * (router_scale / ns as i64);
                objective.backward();
            } else {
                native::backward(&out, &grad)?;
            }
            grad = input.grad().detach();
        }
        let embedding = self.embed(ids);
        native::backward(&embedding, &grad)?;
        Ok((loss, aux, load))
    }
}
pub fn ce(logits: &Tensor, labels: &Tensor) -> Tensor {
    logits
        .to_kind(Kind::Float)
        .reshape([-1, logits.size()[logits.dim() - 1]])
        .cross_entropy_loss(
            &labels.reshape([-1]),
            None::<&Tensor>,
            Reduction::Sum,
            -100,
            0.,
        )
}
pub fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let f = x.to_kind(Kind::Float);
    // The Python training reference casts x twice. Keep separate autograd
    // edges: BF16 rounds each cast's backward contribution before accumulation.
    let second = x.to_kind(Kind::Float);
    (&f * (second.square().mean_dim([-1].as_slice(), true, None) + eps).rsqrt()
        * weight.to_kind(Kind::Float))
    .to_kind(x.kind())
}
pub fn normalize(x: &Tensor) -> Tensor {
    x / x
        .norm_scalaropt_dim(2., [-1], true)
        .clamp_min(1e-12)
        .expand_as(x)
}
pub fn rotary(x: &Tensor, positions: &Tensor, theta: f64) -> Tensor {
    let half = x.size()[x.dim() - 1] / 2;
    let freq = Tensor::pow_scalar(
        theta,
        &(-Tensor::arange(half, (Kind::Float, x.device())) / half),
    );
    let angle = positions.unsqueeze(1).unsqueeze(-1).to_kind(Kind::Float) * freq;
    let cosine = angle.cos().to_kind(x.kind());
    let sine = angle.sin().to_kind(x.kind());
    let a = x.narrow(-1, 0, half);
    let b = x.narrow(-1, half, half);
    Tensor::cat(&[&a * &cosine - &b * &sine, b * cosine + a * sine], -1)
}
