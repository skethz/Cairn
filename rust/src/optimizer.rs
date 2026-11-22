use crate::{model::Model, native};
use anyhow::Result;
use tch::{Kind, Tensor};
pub struct State {
    pub mean: Tensor,
    pub variance: Tensor,
    pub step: Tensor,
}
pub struct AdamW {
    pub states: Vec<State>,
}
impl AdamW {
    pub fn new(m: &Model) -> Self {
        Self {
            states: m
                .params
                .iter()
                .map(|p| State {
                    mean: Tensor::zeros_like(&p.tensor),
                    variance: Tensor::zeros_like(&p.tensor),
                    step: Tensor::zeros([], (Kind::Float, p.tensor.device())),
                })
                .collect(),
        }
    }
    pub fn step(&self, m: &Model, lr: f64) -> Result<()> {
        for decay in [true, false] {
            let indices: Vec<_> = m
                .params
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    let d = p.tensor.dim() >= 2
                        && !["norm", "embedding", "A_log", "dt_bias"]
                            .iter()
                            .any(|n| p.name.contains(n));
                    d == decay
                })
                .map(|(i, _)| i)
                .collect();
            native::adamw(
                &indices
                    .iter()
                    .map(|i| &m.params[*i].tensor)
                    .collect::<Vec<_>>(),
                &indices
                    .iter()
                    .map(|i| &self.states[*i].mean)
                    .collect::<Vec<_>>(),
                &indices
                    .iter()
                    .map(|i| &self.states[*i].variance)
                    .collect::<Vec<_>>(),
                &indices
                    .iter()
                    .map(|i| &self.states[*i].step)
                    .collect::<Vec<_>>(),
                lr,
                if decay { 0.1 } else { 0. },
            )?;
        }
        Ok(())
    }
    pub fn clip(&self, m: &Model, max: f64) -> Result<f64> {
        native::clip(&m.params.iter().map(|p| &p.tensor).collect::<Vec<_>>(), max)
    }
}
