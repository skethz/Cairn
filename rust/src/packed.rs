//! Port of the owned, audited packed inference implementation, including MTP.
//! No original repository, loader, weights, or tokenizer are needed.
use crate::{
    io,
    model::{normalize, rotary},
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};
use tch::{Device, IndexOp, Kind, Tensor};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub hidden_act: String,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: i64,
    pub tie_word_embeddings: bool,
    pub intermediate_size: i64,
    pub gdn_config: Gdn,
    pub mla_config: Mla,
    pub moe_config: Moe,
    pub mtp_config: serde_json::Value,
    pub dense_layer_indices: Vec<i64>,
    pub mla_layer_indices: Vec<i64>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Gdn {
    pub num_heads: i64,
    pub head_dim_k: i64,
    pub head_dim_v: i64,
    pub short_conv_kernel: i64,
    #[serde(default = "yes")]
    pub qk_l2norm: bool,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default)]
    pub chunk_size: i64,
    #[serde(default = "state")]
    pub state_dtype: String,
}
fn yes() -> bool {
    true
}
fn state() -> String {
    "float32".into()
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Mla {
    pub num_heads: i64,
    pub q_lora_rank: i64,
    pub kv_lora_rank: i64,
    pub qk_nope_head_dim: i64,
    pub qk_rope_head_dim: i64,
    pub v_head_dim: i64,
    pub rope_theta: f64,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Groups {
    pub n_group: i64,
    pub topk_group: i64,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Moe {
    pub n_routed_experts: i64,
    pub n_shared_experts: i64,
    pub num_experts_per_tok: i64,
    pub moe_intermediate_size: i64,
    pub group_limited_routing: Groups,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        let g = &self.gdn_config;
        let a = &self.mla_config;
        let m = &self.moe_config;
        for n in [
            self.vocab_size,
            self.hidden_size,
            self.num_hidden_layers,
            self.intermediate_size,
            self.max_position_embeddings,
            g.num_heads,
            g.head_dim_k,
            g.head_dim_v,
            g.short_conv_kernel,
            a.num_heads,
            a.q_lora_rank,
            a.kv_lora_rank,
            a.qk_nope_head_dim,
            a.qk_rope_head_dim,
            a.v_head_dim,
            m.n_routed_experts,
            m.num_experts_per_tok,
            m.moe_intermediate_size,
            m.group_limited_routing.n_group,
            m.group_limited_routing.topk_group,
        ] {
            ensure!(n > 0 && n <= 1_000_000, "Invalid packed dimensions");
        }
        ensure!(
            self.num_hidden_layers <= 4096
                && m.n_routed_experts <= 65536
                && m.moe_intermediate_size <= 65536,
            "Dimension budget exceeded"
        );
        ensure!(
            !self.tie_word_embeddings
                && ["silu", "relu", "gelu"].contains(&self.hidden_act.as_str())
                && g.state_dtype == "float32",
            "Unsupported packed configuration"
        );
        ensure!(
            self.rms_norm_eps > 0.
                && self.rms_norm_eps.is_finite()
                && a.rope_theta > 0.
                && a.rope_theta.is_finite()
                && m.routed_scaling_factor > 0.
                && m.routed_scaling_factor.is_finite(),
            "Invalid scales"
        );
        ensure!(
            g.num_heads * g.head_dim_v == self.hidden_size
                && a.num_heads * a.v_head_dim == self.hidden_size
                && a.qk_rope_head_dim % 2 == 0,
            "Incompatible mixer widths"
        );
        ensure!(
            [
                self.hidden_size,
                a.q_lora_rank,
                a.kv_lora_rank,
                m.moe_intermediate_size
            ]
            .iter()
            .all(|n| n % 16 == 0),
            "Packed width must be divisible by 16"
        );
        let groups = m.group_limited_routing.n_group;
        let keep = m.group_limited_routing.topk_group;
        ensure!(
            m.n_shared_experts == 1
                && m.n_routed_experts % groups == 0
                && m.n_routed_experts / groups >= 2
                && keep <= groups
                && m.num_experts_per_tok <= keep * (m.n_routed_experts / groups),
            "Invalid expert routing"
        );
        for xs in [&self.dense_layer_indices, &self.mla_layer_indices] {
            let unique: std::collections::HashSet<_> = xs.iter().collect();
            ensure!(
                unique.len() == xs.len()
                    && xs.iter().all(|x| *x >= 0 && *x < self.num_hidden_layers),
                "Invalid layer indices"
            );
        }
        let mtp = self
            .mtp_config
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Invalid MTP"))?;
        ensure!(
            mtp.keys().all(
                |k| ["depth", "layer_index_in_checkpoint", "loss_weight"].contains(&k.as_str())
            ) && self.mtp_config["depth"].as_i64().unwrap_or(1) == 1,
            "Unsupported MTP"
        );
        Ok(())
    }
    pub fn constructor_bytes(&self) -> Result<u64> {
        self.validate()?;
        let h = self.hidden_size as u128;
        let g = &self.gdn_config;
        let a = &self.mla_config;
        let m = &self.moe_config;
        let linear = |i: u128, o: u128, packed: bool| {
            if packed {
                o * (i / 2 + i / 16) + 8
            } else {
                4 * i * o
            }
        };
        let mixer = |mla: bool, packed: bool| {
            if mla {
                let pairs = [
                    (h, a.q_lora_rank as u128),
                    (
                        a.q_lora_rank as u128,
                        (a.num_heads * (a.qk_nope_head_dim + a.qk_rope_head_dim)) as u128,
                    ),
                    (h, (a.kv_lora_rank + a.qk_rope_head_dim) as u128),
                    (
                        a.kv_lora_rank as u128,
                        (a.num_heads * (a.qk_nope_head_dim + a.v_head_dim)) as u128,
                    ),
                    (h, h),
                    (h, h),
                ];
                pairs
                    .iter()
                    .map(|(i, o)| linear(*i, *o, packed))
                    .sum::<u128>()
                    + 4 * (a.q_lora_rank + a.kv_lora_rank) as u128
            } else {
                let key = (g.num_heads * g.head_dim_k) as u128;
                let value = (g.num_heads * g.head_dim_v) as u128;
                [key, key, value, value, value]
                    .iter()
                    .map(|w| linear(h, *w, packed))
                    .sum::<u128>()
                    + 8 * h * g.num_heads as u128
                    + 4 * (2 * key + value)
                        * (g.short_conv_kernel as u128 + u128::from(g.conv_bias))
                    + 8 * g.num_heads as u128
                    + 4 * g.head_dim_v as u128
            }
        };
        let dense = 12 * h * self.intermediate_size as u128;
        let expert = 2 * linear(h, m.moe_intermediate_size as u128, true)
            + linear(m.moe_intermediate_size as u128, h, true);
        let moe = 4 * h * m.n_routed_experts as u128
            + 4 * m.n_routed_experts as u128
            + (m.n_routed_experts as u128 + 1) * expert;
        let mut total = 8 * self.vocab_size as u128 * h + 4 * h;
        for i in 0..self.num_hidden_layers {
            let d = self.dense_layer_indices.contains(&i);
            total += mixer(self.mla_layer_indices.contains(&i), !d)
                + 8 * h
                + if d { dense } else { moe };
        }
        Ok(u64::try_from(
            total + mixer(true, false) + dense + 8 * h * h + 16 * h,
        )?)
    }
}
pub fn unpack(p: &Tensor) -> Tensor {
    let low = p.bitwise_and(15).to_kind(Kind::Int64);
    let high = p
        .bitwise_right_shift_tensor_scalar(4)
        .bitwise_and(15)
        .to_kind(Kind::Int64);
    let codes = Tensor::stack(&[low, high], -1);
    let sign = codes.bitwise_and(8).ne(0);
    let lut = Tensor::from_slice(&[0f32, 0.5, 1., 1.5, 2., 3., 4., 6.]).to_device(p.device());
    let shape = codes.size();
    let values = lut
        .index_select(0, &codes.bitwise_and(7).flatten(0, -1))
        .reshape(&shape);
    let sign = Tensor::full_like(&values, -1.).where_self(&sign, &Tensor::ones_like(&values));
    (values * sign).flatten(-2, -1)
}
pub struct Packed {
    pub config: Config,
    weights: HashMap<String, Tensor>,
}
impl Packed {
    pub fn load(c: Config, path: &Path, device: Device, max: u64) -> Result<Self> {
        ensure!(
            max > 0 && max <= 2 * 1024_u64.pow(3) && c.constructor_bytes()? <= max,
            "Packed model allocation budget exceeded"
        );
        let weights = io::tensors(path, max)?
            .into_iter()
            .map(|(n, t)| (n, t.to_device(device)))
            .collect();
        let m = Self { config: c, weights };
        m.validate_weights()?;
        Ok(m)
    }
    fn w(&self, n: &str) -> &Tensor {
        self.weights
            .get(n)
            .unwrap_or_else(|| panic!("Missing packed tensor {n}"))
    }
    fn validate_weights(&self) -> Result<()> {
        // Exact manifest generated without tensor allocations; extra keys rejected.
        let expected = self.layout();
        ensure!(
            expected.len() == self.weights.len(),
            "Packed tensor manifest mismatch"
        );
        for (name, shape, kind) in expected {
            let t = self
                .weights
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("Missing tensor {name}"))?;
            ensure!(t.size() == shape, "Packed tensor shape mismatch: {name}");
            ensure!(
                t.kind() == kind || (kind == Kind::BFloat16 && t.kind() == Kind::Float),
                "Packed tensor dtype mismatch: {name}"
            );
        }
        Ok(())
    }
    fn layout(&self) -> Vec<(String, Vec<i64>, Kind)> {
        let mut v = Vec::new();
        let c = &self.config;
        let h = c.hidden_size;
        let g = &c.gdn_config;
        let a = &c.mla_config;
        let m = &c.moe_config;
        fn norm(v: &mut Vec<(String, Vec<i64>, Kind)>, n: &str, d: i64) {
            v.push((format!("{n}.weight"), vec![d], Kind::BFloat16));
        }
        fn linear(v: &mut Vec<(String, Vec<i64>, Kind)>, n: &str, i: i64, o: i64, p: bool) {
            if p {
                v.push((format!("{n}.weight"), vec![o, i / 2], Kind::Uint8));
                v.push((
                    format!("{n}.weight_scale"),
                    vec![o, i / 16],
                    Kind::Float8e4m3fn,
                ));
                for f in ["weight_scale_2", "input_scale"] {
                    v.push((format!("{n}.{f}"), vec![], Kind::Float));
                }
            } else {
                v.push((format!("{n}.weight"), vec![o, i], Kind::BFloat16));
            }
        }
        let mixer = |v: &mut Vec<_>, n: &str, mla: bool, p: bool| {
            if mla {
                for (s, i, o) in [
                    ("q_a_proj", h, a.q_lora_rank),
                    (
                        "q_b_proj",
                        a.q_lora_rank,
                        a.num_heads * (a.qk_nope_head_dim + a.qk_rope_head_dim),
                    ),
                    ("kv_a_proj_with_mqa", h, a.kv_lora_rank + a.qk_rope_head_dim),
                    (
                        "kv_b_proj",
                        a.kv_lora_rank,
                        a.num_heads * (a.qk_nope_head_dim + a.v_head_dim),
                    ),
                    ("o_gate_proj", h, h),
                    ("o_proj", h, h),
                ] {
                    linear(v, &format!("{n}.{s}"), i, o, p);
                }
                norm(v, &format!("{n}.q_a_layernorm"), a.q_lora_rank);
                norm(v, &format!("{n}.kv_a_layernorm"), a.kv_lora_rank);
            } else {
                for (s, o) in [
                    ("q", g.num_heads * g.head_dim_k),
                    ("k", g.num_heads * g.head_dim_k),
                    ("v", h),
                    ("g", h),
                    ("o", h),
                ] {
                    linear(v, &format!("{n}.{s}_proj"), h, o, p);
                }
                for s in ["a", "b"] {
                    linear(v, &format!("{n}.{s}_proj"), h, g.num_heads, false);
                }
                for (s, w) in [
                    ("q", g.num_heads * g.head_dim_k),
                    ("k", g.num_heads * g.head_dim_k),
                    ("v", h),
                ] {
                    v.push((
                        format!("{n}.{s}_conv1d.weight"),
                        vec![w, 1, g.short_conv_kernel],
                        Kind::BFloat16,
                    ));
                    if g.conv_bias {
                        v.push((format!("{n}.{s}_conv1d.bias"), vec![w], Kind::BFloat16));
                    }
                }
                for s in ["A_log", "dt_bias"] {
                    v.push((format!("{n}.{s}"), vec![g.num_heads], Kind::Float));
                }
                norm(v, &format!("{n}.o_norm"), g.head_dim_v);
            }
        };
        let mlp = |v: &mut Vec<_>, n: &str, w: i64, p: bool| {
            linear(v, &format!("{n}.gate_proj"), h, w, p);
            linear(v, &format!("{n}.up_proj"), h, w, p);
            linear(v, &format!("{n}.down_proj"), w, h, p);
        };
        linear(&mut v, "model.embed_tokens", h, c.vocab_size, false);
        for i in 0..c.num_hidden_layers {
            let n = format!("model.layers.{i}");
            let dense = c.dense_layer_indices.contains(&i);
            mixer(
                &mut v,
                &format!("{n}.self_attn"),
                c.mla_layer_indices.contains(&i),
                !dense,
            );
            for s in ["input_layernorm", "post_attention_layernorm"] {
                norm(&mut v, &format!("{n}.{s}"), h);
            }
            if dense {
                mlp(&mut v, &format!("{n}.mlp"), c.intermediate_size, false);
            } else {
                linear(
                    &mut v,
                    &format!("{n}.mlp.gate"),
                    h,
                    m.n_routed_experts,
                    false,
                );
                v.push((
                    format!("{n}.mlp.gate.e_score_correction_bias"),
                    vec![m.n_routed_experts],
                    Kind::Float,
                ));
                mlp(
                    &mut v,
                    &format!("{n}.mlp.shared_experts"),
                    m.moe_intermediate_size,
                    true,
                );
                for e in 0..m.n_routed_experts {
                    mlp(
                        &mut v,
                        &format!("{n}.mlp.experts.{e}"),
                        m.moe_intermediate_size,
                        true,
                    );
                }
            }
        }
        let n = format!("model.layers.{}", c.num_hidden_layers);
        for s in [
            "enorm",
            "hnorm",
            "input_layernorm",
            "post_attention_layernorm",
        ] {
            norm(&mut v, &format!("{n}.{s}"), h);
        }
        linear(&mut v, &format!("{n}.eh_proj"), 2 * h, h, false);
        mixer(&mut v, &format!("{n}.self_attn"), true, false);
        mlp(&mut v, &format!("{n}.mlp"), c.intermediate_size, false);
        norm(&mut v, "model.norm", h);
        linear(&mut v, "lm_head", h, c.vocab_size, false);
        v
    }
    pub fn linear(&self, n: &str, x: &Tensor) -> Tensor {
        let w = self.w(&format!("{n}.weight"));
        if w.kind() != Kind::Uint8 {
            return x.linear(w, None::<&Tensor>);
        }
        let decoded = unpack(w).to_kind(Kind::Float);
        let decoded = decoded.reshape([w.size()[0], -1, 16])
            * self
                .w(&format!("{n}.weight_scale"))
                .to_kind(Kind::Float)
                .unsqueeze(-1);
        let decoded = (decoded.reshape([w.size()[0], -1]) * self.w(&format!("{n}.weight_scale_2")))
            .to_kind(Kind::BFloat16);
        let input_scale = self.w(&format!("{n}.input_scale"));
        if input_scale.double_value(&[]) != 0. {
            let shape = x.size();
            let mut blocks_shape = shape[..shape.len() - 1].to_vec();
            blocks_shape.extend([-1, 16]);
            let blocks =
                (x.to_kind(Kind::Float) / input_scale.to_kind(Kind::Float)).reshape(&blocks_shape);
            let scale = (blocks.abs().amax([-1], true) / 6.)
                .clamp_min(1e-8)
                .to_kind(Kind::Float8e4m3fn)
                .to_kind(Kind::Float)
                .clamp_min(1e-8);
            let bounds = Tensor::from_slice(&[
                0.25f32,
                f32::from_bits(0.75f32.to_bits() - 1),
                1.25,
                f32::from_bits(1.75f32.to_bits() - 1),
                2.5,
                f32::from_bits(3.5f32.to_bits() - 1),
                5.,
            ])
            .to_device(x.device());
            let codes = (blocks.abs() / &scale).bucketize(&bounds, false, false);
            let lut =
                Tensor::from_slice(&[0f32, 0.5, 1., 1.5, 2., 3., 4., 6.]).to_device(x.device());
            let quantized = lut
                .index_select(0, &codes.flatten(0, -1))
                .reshape(codes.size())
                * blocks.sign()
                * scale;
            quantized
                .to_kind(decoded.kind())
                .reshape(&shape)
                .linear(&decoded, None::<&Tensor>)
                * input_scale.to_kind(decoded.kind())
        } else {
            x.to_kind(decoded.kind()).linear(&decoded, None::<&Tensor>)
        }
    }
    fn norm(&self, n: &str, x: &Tensor) -> Tensor {
        let f = x.to_kind(Kind::Float);
        (&f * (f.square().mean_dim([-1].as_slice(), true, None) + self.config.rms_norm_eps).rsqrt()
            * self.w(&format!("{n}.weight")).to_kind(Kind::Float))
        .to_kind(x.kind())
    }
    fn mlp(&self, n: &str, x: &Tensor) -> Tensor {
        let gate = self.linear(&format!("{n}.gate_proj"), x);
        let gate = match self.config.hidden_act.as_str() {
            "relu" => gate.relu(),
            "gelu" => gate.gelu("none"),
            _ => gate.silu(),
        };
        self.linear(
            &format!("{n}.down_proj"),
            &(gate * self.linear(&format!("{n}.up_proj"), x)),
        )
    }
    fn delta(&self, n: &str, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let g = &self.config.gdn_config;
        let (b, t, h, k, v) = (
            x.size()[0],
            x.size()[1],
            g.num_heads,
            g.head_dim_k,
            g.head_dim_v,
        );
        let conv = |s: &str, dim: i64| {
            let p = self
                .linear(&format!("{n}.{s}_proj"), x)
                .view([b, t, h, dim]);
            let w = self.w(&format!("{n}.{s}_conv1d.weight"));
            let bias = self.weights.get(&format!("{n}.{s}_conv1d.bias"));
            p.reshape([b, t, -1])
                .transpose(1, 2)
                .constant_pad_nd([g.short_conv_kernel - 1, 0])
                .conv1d(w, bias, [1], [0], [1], h * dim)
                .transpose(1, 2)
                .reshape([b, t, h, dim])
                .silu()
        };
        let mut q = conv("q", k);
        let mut key = conv("k", k);
        let value = conv("v", v).permute([2, 0, 1, 3]);
        if g.qk_l2norm {
            q = normalize(&q);
            key = normalize(&key);
        }
        let q = q.permute([2, 0, 1, 3]);
        let key = key.permute([2, 0, 1, 3]);
        let alpha = (-self.w(&format!("{n}.A_log")).exp()
            * (self.linear(&format!("{n}.a_proj"), x) + self.w(&format!("{n}.dt_bias")))
                .softplus())
        .exp()
        .permute([2, 0, 1]);
        let mut beta = self
            .linear(&format!("{n}.b_proj"), x)
            .sigmoid()
            .permute([2, 0, 1]);
        if let Some(mask) = mask {
            let kind = beta.kind();
            beta = beta * mask.to_kind(kind).unsqueeze(0);
        }
        let mut state = Tensor::zeros([h, b, v, k], (Kind::Float, x.device()));
        let mut outputs = Vec::new();
        for pos in 0..t {
            let key = key.i((.., .., pos)).to_kind(Kind::Float);
            let val = value.i((.., .., pos)).to_kind(Kind::Float);
            let decay = alpha.narrow(2, pos, 1).unsqueeze(-1);
            let write = beta.narrow(2, pos, 1).unsqueeze(-1);
            state = &decay
                * (&state - &write * state.matmul(&key.unsqueeze(-1)).matmul(&key.unsqueeze(-2)));
            state = state + &write * val.unsqueeze(-1) * key.unsqueeze(-2);
            outputs.push(
                state
                    .matmul(&q.i((.., .., pos)).to_kind(Kind::Float).unsqueeze(-1))
                    .squeeze_dim(-1),
            );
        }
        let out = self
            .norm(&format!("{n}.o_norm"), &Tensor::stack(&outputs, 2))
            .permute([1, 2, 0, 3])
            .reshape([b, t, h * v])
            .to_kind(x.kind());
        self.linear(
            &format!("{n}.o_proj"),
            &(out * self.linear(&format!("{n}.g_proj"), x).silu()),
        )
    }
    fn latent(
        &self,
        n: &str,
        x: &Tensor,
        positions: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Tensor {
        let a = &self.config.mla_config;
        let (b, t, h) = (x.size()[0], x.size()[1], a.num_heads);
        let positions = positions
            .map(Tensor::shallow_clone)
            .unwrap_or_else(|| Tensor::arange(t, (Kind::Float, x.device())).expand([b, t], false));
        let q = self
            .linear(
                &format!("{n}.q_b_proj"),
                &self.norm(
                    &format!("{n}.q_a_layernorm"),
                    &self.linear(&format!("{n}.q_a_proj"), x),
                ),
            )
            .view([b, t, h, -1])
            .permute([0, 2, 1, 3]);
        let compressed = self.linear(&format!("{n}.kv_a_proj_with_mqa"), x);
        let kv = self
            .linear(
                &format!("{n}.kv_b_proj"),
                &self.norm(
                    &format!("{n}.kv_a_layernorm"),
                    &compressed.narrow(-1, 0, a.kv_lora_rank),
                ),
            )
            .view([b, t, h, -1])
            .permute([0, 2, 1, 3]);
        let kr = compressed
            .narrow(-1, a.kv_lora_rank, a.qk_rope_head_dim)
            .unsqueeze(1)
            .expand([-1, h, -1, -1], false);
        let q = Tensor::cat(
            &[
                q.narrow(-1, 0, a.qk_nope_head_dim),
                rotary(
                    &q.narrow(-1, a.qk_nope_head_dim, a.qk_rope_head_dim),
                    &positions,
                    a.rope_theta,
                ),
            ],
            -1,
        );
        let k = Tensor::cat(
            &[
                kv.narrow(-1, 0, a.qk_nope_head_dim),
                rotary(&kr, &positions, a.rope_theta),
            ],
            -1,
        );
        let v = kv.narrow(-1, a.qk_nope_head_dim, a.v_head_dim);
        let out = if let Some(mask) = mask {
            let keep = mask.to_kind(Kind::Bool).unsqueeze(1).unsqueeze(1);
            let attn = Tensor::zeros([b, 1, 1, t], (q.kind(), q.device()))
                .masked_fill(&keep.logical_not(), f64::NEG_INFINITY)
                + Tensor::full([t, t], f64::NEG_INFINITY, (q.kind(), q.device())).triu(1);
            Tensor::scaled_dot_product_attention(&q, &k, &v, Some(&attn), 0., false, None, false)
        } else {
            Tensor::scaled_dot_product_attention(&q, &k, &v, None::<&Tensor>, 0., true, None, false)
        };
        self.linear(
            &format!("{n}.o_proj"),
            &(out.transpose(1, 2).reshape([b, t, -1])
                * self.linear(&format!("{n}.o_gate_proj"), x).sigmoid()),
        )
    }
    fn moe(&self, n: &str, x: &Tensor) -> Tensor {
        let m = &self.config.moe_config;
        let g = &m.group_limited_routing;
        let (b, t) = (x.size()[0], x.size()[1]);
        let scores = x
            .to_kind(Kind::Float)
            .linear(
                &self.w(&format!("{n}.gate.weight")).to_kind(Kind::Float),
                None::<&Tensor>,
            )
            .sigmoid();
        let corrected = &scores + self.w(&format!("{n}.gate.e_score_correction_bias"));
        let group = corrected
            .view([b, t, g.n_group, -1])
            .topk(2, -1, true, true)
            .0
            .sum_dim_intlist([-1].as_slice(), false, None);
        let retained = group.topk(g.topk_group, -1, true, true).1;
        let membership = Tensor::arange(m.n_routed_experts, (Kind::Int64, x.device()))
            .floor_divide_scalar(m.n_routed_experts / g.n_group);
        let candidates = retained
            .unsqueeze(-1)
            .eq_tensor(&membership.view([1, 1, -1]))
            .any_dim(2, false);
        let selected = corrected
            .masked_fill(&candidates.logical_not(), f64::NEG_INFINITY)
            .topk(m.num_experts_per_tok, -1, true, true)
            .1;
        let mut weights = scores.gather(-1, &selected, false);
        if m.norm_topk_prob {
            weights = &weights / weights.sum_dim_intlist([-1].as_slice(), true, None)
                * m.routed_scaling_factor;
        }
        let flat = x.view([-1, x.size()[2]]);
        let selected = selected.view([-1, m.num_experts_per_tok]);
        let weights = weights.view([-1, m.num_experts_per_tok]).to_kind(x.kind());
        let mut output = Tensor::zeros_like(&flat);
        for e in 0..m.n_routed_experts {
            let hits = selected.eq(e);
            let rows = hits.any_dim(-1, false).nonzero().select(1, 0);
            if rows.numel() == 0 {
                continue;
            }
            let w = weights
                .index_select(0, &rows)
                .masked_select(&hits.index_select(0, &rows))
                .unsqueeze(-1);
            let _ = output.index_add_(
                0,
                &rows,
                &(self.mlp(&format!("{n}.experts.{e}"), &flat.index_select(0, &rows)) * w),
            );
        }
        self.mlp(&format!("{n}.shared_experts"), x) + output.view([b, t, -1])
    }
    pub fn logits(
        &self,
        ids: &Tensor,
        positions: Option<&Tensor>,
        mask: Option<&Tensor>,
        draft: bool,
    ) -> Result<Tensor> {
        ensure!(
            ids.kind() == Kind::Int64
                && ids.dim() == 2
                && ids.size()[0] <= 16
                && ids.size()[1] > 0
                && ids.size()[1] <= self.config.max_position_embeddings.min(4096),
            "Invalid local inference input"
        );
        ensure!(
            ids.min().int64_value(&[]) >= 0 && ids.max().int64_value(&[]) < self.config.vocab_size,
            "Invalid token IDs"
        );
        if let Some(p) = positions {
            ensure!(p.size() == ids.size(), "Position shape mismatch");
        }
        if let Some(m) = mask {
            ensure!(m.size() == ids.size(), "Mask shape mismatch");
        }
        let embedding =
            Tensor::embedding(self.w("model.embed_tokens.weight"), ids, -1, false, false);
        let mut x = embedding.shallow_clone();
        for i in 0..self.config.num_hidden_layers {
            let n = format!("model.layers.{i}");
            let norm = self.norm(&format!("{n}.input_layernorm"), &x);
            let mix = if self.config.mla_layer_indices.contains(&i) {
                self.latent(&format!("{n}.self_attn"), &norm, positions, mask)
            } else {
                self.delta(&format!("{n}.self_attn"), &norm, mask)
            };
            x = x + mix;
            let norm = self.norm(&format!("{n}.post_attention_layernorm"), &x);
            let f = if self.config.dense_layer_indices.contains(&i) {
                self.mlp(&format!("{n}.mlp"), &norm)
            } else {
                self.moe(&format!("{n}.mlp"), &norm)
            };
            x = x + f;
        }
        if draft {
            ensure!(ids.size()[1] >= 2, "Draft needs at least two input tokens");
            let n = format!("model.layers.{}", self.config.num_hidden_layers);
            let length = ids.size()[1] - 1;
            let e = self.norm(&format!("{n}.enorm"), &embedding.narrow(1, 1, length));
            let h = self.norm(&format!("{n}.hnorm"), &x.narrow(1, 0, length));
            x = self.linear(&format!("{n}.eh_proj"), &Tensor::cat(&[e, h], -1));
            let mix = self.latent(
                &format!("{n}.self_attn"),
                &self.norm(&format!("{n}.input_layernorm"), &x),
                None,
                None,
            );
            x = x + mix;
            let mlp = self.mlp(
                &format!("{n}.mlp"),
                &self.norm(&format!("{n}.post_attention_layernorm"), &x),
            );
            x = x + mlp;
        }
        Ok(self.linear("lm_head", &self.norm("model.norm", &x)))
    }
}
