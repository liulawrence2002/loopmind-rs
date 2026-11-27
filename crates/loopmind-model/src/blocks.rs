//! Pre-norm attention, SwiGLU, and top-k mixture of experts.
//!
//! Linear maps store Burn's layout: weight `[d_input, d_output]`, `y = x @ weight`, no bias.
//! The PyTorch reference uses that same layout rather than `nn.Linear`.

use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::{Bool, Device, Int, Tensor};

use crate::config::LoopMindConfig;

const CAUSAL_FILL: f32 = -1.0e9;
const TOPK_EPS: f32 = 1.0e-9;

pub(crate) fn linear(d_input: usize, d_output: usize, device: &Device) -> Linear {
    LinearConfig::new(d_input, d_output)
        .with_bias(false)
        .init(device)
}

pub(crate) fn rms_norm(d_model: usize, epsilon: f64, device: &Device) -> RmsNorm {
    RmsNormConfig::new(d_model)
        .with_epsilon(epsilon)
        .init(device)
}

/// Grouped-query attention with RoPE and a causal mask.
#[derive(burn::module::Module, Debug)]
pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    #[module(skip)]
    pub n_q_heads: usize,
    #[module(skip)]
    pub n_kv_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub rope_theta: f64,
}

impl Attention {
    pub(crate) fn new(config: &LoopMindConfig, device: &Device) -> Self {
        let head_dim = config.head_dim();
        let q_width = config.n_q_heads * head_dim;
        let kv_width = config.n_kv_heads * head_dim;
        Self {
            q: linear(config.d_model, q_width, device),
            k: linear(config.d_model, kv_width, device),
            v: linear(config.d_model, kv_width, device),
            o: linear(q_width, config.d_model, device),
            n_q_heads: config.n_q_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim,
            rope_theta: config.rope_theta,
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>, ctx: &AttnContext) -> Tensor<3> {
        let [batch, seq, _] = x.dims();
        let q = self.project_heads(self.q.forward(x.clone()), self.n_q_heads, batch, seq);
        let k = self.project_heads(self.k.forward(x.clone()), self.n_kv_heads, batch, seq);
        let v = self.project_heads(self.v.forward(x), self.n_kv_heads, batch, seq);
        let q = apply_rope(q, &ctx.cos, &ctx.sin);
        let k = apply_rope(k, &ctx.cos, &ctx.sin);
        let n_rep = self.n_q_heads / self.n_kv_heads;
        let k = repeat_kv(k, n_rep);
        let v = repeat_kv(v, n_rep);
        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;
        let scores = scores.mask_fill(ctx.mask.clone(), CAUSAL_FILL);
        let probs = softmax(scores, 3);
        let context =
            probs
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([batch, seq, self.n_q_heads * self.head_dim]);
        self.o.forward(context)
    }

    fn project_heads(
        &self,
        projected: Tensor<3>,
        heads: usize,
        batch: usize,
        seq: usize,
    ) -> Tensor<4> {
        projected
            .reshape([batch, seq, heads, self.head_dim])
            .swap_dims(1, 2)
    }
}

/// RoPE tables and the causal mask for one `[batch, seq]` forward.
///
/// Every attention layer and every loop pass sees the same positions, so the
/// tables are built once per forward instead of once per attention call.
#[derive(Debug, Clone)]
pub(crate) struct AttnContext {
    cos: Tensor<4>,
    sin: Tensor<4>,
    mask: Tensor<4, Bool>,
}

impl AttnContext {
    pub(crate) fn new(config: &LoopMindConfig, batch: usize, seq: usize, device: &Device) -> Self {
        let (cos, sin) = rope_tables(seq, config.head_dim(), config.rope_theta, device);
        Self {
            cos,
            sin,
            mask: causal_mask(batch, config.n_q_heads, seq, device),
        }
    }

    /// The same tables without autodiff, for passes that are not backpropagated.
    pub(crate) fn inner(self) -> Self {
        Self {
            cos: self.cos.inner(),
            sin: self.sin.inner(),
            mask: self.mask.inner(),
        }
    }

    /// Match the dtype of the activations, so a bf16 model does not multiply by f32 tables.
    pub(crate) fn cast(self, dtype: burn::tensor::DType) -> Self {
        Self {
            cos: self.cos.cast(dtype),
            sin: self.sin.cast(dtype),
            mask: self.mask,
        }
    }
}

fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }
    let [batch, kv, seq, head] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .repeat_dim(2, n_rep)
        .reshape([batch, kv * n_rep, seq, head])
}

fn causal_mask(batch: usize, heads: usize, seq: usize, device: &Device) -> Tensor<4, Bool> {
    // `true` marks positions j > i, which the attention scores replace with a large negative.
    Tensor::<2, Bool>::tril_mask([seq, seq], 0, device)
        .unsqueeze_dims::<4>(&[0, 1])
        .repeat(&[batch, heads, 1, 1])
}

/// GPT-NeoX / `rotate_half` RoPE. `cos` and `sin` are `[1, 1, seq, head_dim]`.
fn rope_tables(seq: usize, head_dim: usize, theta: f64, device: &Device) -> (Tensor<4>, Tensor<4>) {
    let half = head_dim / 2;
    let mut angles = Vec::with_capacity(seq * half);
    for pos in 0..seq {
        for i in 0..half {
            let exponent = (2 * i) as f64 / head_dim as f64;
            let inv_freq = theta.powf(-exponent) as f32;
            angles.push(pos as f32 * inv_freq);
        }
    }
    let angles = Tensor::<2>::from_data(burn::tensor::TensorData::new(angles, [seq, half]), device);
    let cos_half = angles.clone().cos();
    let sin_half = angles.sin();
    let cos = Tensor::cat(vec![cos_half.clone(), cos_half], 1);
    let sin = Tensor::cat(vec![sin_half.clone(), sin_half], 1);
    (
        cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0),
        sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0),
    )
}

fn apply_rope(x: Tensor<4>, cos: &Tensor<4>, sin: &Tensor<4>) -> Tensor<4> {
    x.clone() * cos.clone() + rotate_half(x) * sin.clone()
}

fn rotate_half(x: Tensor<4>) -> Tensor<4> {
    let half = x.dims()[3] / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.narrow(3, half, half);
    Tensor::cat(vec![-x2, x1], 3)
}

/// Dense SwiGLU: `down(silu(gate(x)) * up(x))`.
#[derive(burn::module::Module, Debug)]
pub struct SwiGlu {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl SwiGlu {
    pub(crate) fn new(d_model: usize, hidden: usize, device: &Device) -> Self {
        Self {
            gate: linear(d_model, hidden, device),
            up: linear(d_model, hidden, device),
            down: linear(hidden, d_model, device),
        }
    }

    pub(crate) fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let gated = silu(self.gate.forward(x.clone())) * self.up.forward(x);
        self.down.forward(gated)
    }
}

/// Top-k SwiGLU experts with a softmax router and a Switch-style balance loss.
#[derive(burn::module::Module, Debug)]
pub struct Mixture {
    pub router: Linear,
    pub experts: Vec<SwiGlu>,
    #[module(skip)]
    pub top_k: usize,
    #[module(skip)]
    pub norm_topk: bool,
}

impl Mixture {
    pub fn new(config: &LoopMindConfig, device: &Device) -> Self {
        let experts = (0..config.n_experts)
            .map(|_| SwiGlu::new(config.d_model, config.expert_intermediate_size, device))
            .collect();
        Self {
            router: linear(config.d_model, config.n_experts, device),
            experts,
            top_k: config.experts_per_token,
            norm_topk: true,
        }
    }

    /// `x` is `[batch, seq, d_model]`. Optional routes are `[batch, seq, top_k]` expert ids.
    pub fn forward(
        &self,
        x: Tensor<3>,
        force_routes: Option<Tensor<3, Int>>,
    ) -> (Tensor<3>, Tensor<1>, Tensor<1>) {
        self.forward_fused(x, force_routes, &self.fuse())
    }

    /// The experts' weights side by side, for [`Mixture::forward_fused`].
    ///
    /// The core is shared across loop passes, so a forward builds this once per
    /// layer and every pass reuses it. Gradients flow back through the concat to
    /// each expert's own `Linear`, so the checkpoint layout does not change.
    pub(crate) fn fuse(&self) -> FusedExperts {
        let gate = Tensor::cat(
            self.experts
                .iter()
                .map(|expert| expert.gate.weight.val())
                .collect(),
            1,
        );
        let up = Tensor::cat(
            self.experts
                .iter()
                .map(|expert| expert.up.weight.val())
                .collect(),
            1,
        );
        let down = Tensor::cat(
            self.experts
                .iter()
                .map(|expert| expert.down.weight.val())
                .collect(),
            0,
        );
        FusedExperts { gate, up, down }
    }

    /// Every expert evaluated as one wide SwiGLU.
    ///
    /// `down_e(h_e) * g_e` summed over experts equals `concat_e(h_e * g_e) @ concat_e(down_e)`,
    /// so three `[tokens, n_experts * width]` matmuls replace `3 * n_experts` small ones.
    pub(crate) fn forward_fused(
        &self,
        x: Tensor<3>,
        force_routes: Option<Tensor<3, Int>>,
        fused: &FusedExperts,
    ) -> (Tensor<3>, Tensor<1>, Tensor<1>) {
        let [batch, seq, d_model] = x.dims();
        let tokens = batch * seq;
        let flat = x.reshape([tokens, d_model]);
        let (indices, probs, gates) = self.route(flat.clone(), force_routes);
        let n_experts = self.experts.len();
        let width = fused.gate.dims()[1] / n_experts;
        let hidden = silu(flat.clone().matmul(fused.gate.clone())) * flat.matmul(fused.up.clone());
        let scale = gates
            .cast(hidden.dtype())
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, width)
            .reshape([tokens, n_experts * width]);
        let mixed = (hidden * scale).matmul(fused.down.clone());
        let slots = slot_fraction(&indices, n_experts, self.top_k);
        let aux = balance_loss(&probs, &slots, n_experts);
        (mixed.reshape([batch, seq, d_model]), aux, slots)
    }

    /// Top-k ids, router probabilities, and the `[tokens, n_experts]` gate matrix.
    fn route(
        &self,
        flat: Tensor<2>,
        force_routes: Option<Tensor<3, Int>>,
    ) -> (Tensor<2, Int>, Tensor<2>, Tensor<2>) {
        let tokens = flat.dims()[0];
        let n_experts = self.experts.len();
        let logits = self.router.forward(flat);
        let probs = softmax(logits, 1);
        let indices = match force_routes {
            Some(routes) => routes.reshape([tokens, self.top_k]),
            None => probs.clone().topk_with_indices(self.top_k, 1).1,
        };
        let weights = probs.clone().gather(1, indices.clone());
        let weights = if self.norm_topk {
            let denom = weights.clone().sum_dim(1).clamp_min(TOPK_EPS);
            weights / denom
        } else {
            weights
        };
        let gates = route_gates(&indices, &weights, n_experts);
        (indices, probs, gates)
    }

    /// The M1 per-expert loop. Tests check the fused path against it.
    #[cfg(test)]
    pub(crate) fn forward_unfused(
        &self,
        x: Tensor<3>,
        force_routes: Option<Tensor<3, Int>>,
    ) -> (Tensor<3>, Tensor<1>, Tensor<1>) {
        let [batch, seq, d_model] = x.dims();
        let tokens = batch * seq;
        let flat = x.reshape([tokens, d_model]);
        let (indices, probs, gates) = self.route(flat.clone(), force_routes);
        let n_experts = self.experts.len();
        let mut mixed = Tensor::<2>::zeros([tokens, d_model], &flat.device());
        for (expert_id, expert) in self.experts.iter().enumerate() {
            let column = gates.clone().narrow(1, expert_id, 1);
            mixed = mixed + expert.forward(flat.clone()) * column;
        }
        let slots = slot_fraction(&indices, n_experts, self.top_k);
        let aux = balance_loss(&probs, &slots, n_experts);
        (mixed.reshape([batch, seq, d_model]), aux, slots)
    }
}

/// Concatenated expert weights: gate and up are `[d_model, n_experts * width]`,
/// down is `[n_experts * width, d_model]`.
#[derive(Debug, Clone)]
pub(crate) struct FusedExperts {
    gate: Tensor<2>,
    up: Tensor<2>,
    down: Tensor<2>,
}

/// Per-token expert coefficients, shape `[tokens, n_experts]`.
fn route_gates(indices: &Tensor<2, Int>, weights: &Tensor<2>, n_experts: usize) -> Tensor<2> {
    let tokens = indices.dims()[0];
    // Int ids become f32. Match the router's dtype so a bf16 model does not mix the two.
    let assignment = indices
        .clone()
        .float()
        .cast(weights.dtype())
        .one_hot::<3>(n_experts);
    let weighted = assignment * weights.clone().unsqueeze_dim::<3>(2);
    weighted.sum_dim(1).reshape([tokens, n_experts])
}

/// Fraction of top-k slots assigned to each expert.
fn slot_fraction(indices: &Tensor<2, Int>, n_experts: usize, top_k: usize) -> Tensor<1> {
    let tokens = indices.dims()[0] as f32;
    let assignment = indices.clone().float().one_hot::<3>(n_experts);
    assignment.sum_dim(1).sum_dim(0).reshape([n_experts]) / (tokens * top_k as f32)
}

/// `n_experts * sum_i f_i P_i`, where `f` is the fraction of top-k slots and `P` is the mean router probability.
fn balance_loss(probs: &Tensor<2>, slot_fraction: &Tensor<1>, n_experts: usize) -> Tensor<1> {
    let mean_prob = probs.clone().mean_dim(0).reshape([n_experts]);
    (slot_fraction.clone().cast(mean_prob.dtype()) * mean_prob).sum() * (n_experts as f32)
}

/// Pre-norm transformer layer with a dense SwiGLU.
#[derive(burn::module::Module, Debug)]
pub struct DenseBlock {
    pub attn_norm: RmsNorm,
    pub attn: Attention,
    pub ffn_norm: RmsNorm,
    pub ffn: SwiGlu,
}

impl DenseBlock {
    pub(crate) fn new(config: &LoopMindConfig, device: &Device) -> Self {
        Self {
            attn_norm: rms_norm(config.d_model, config.rms_norm_eps, device),
            attn: Attention::new(config, device),
            ffn_norm: rms_norm(config.d_model, config.rms_norm_eps, device),
            ffn: SwiGlu::new(config.d_model, config.dense_intermediate_size, device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>, ctx: &AttnContext) -> Tensor<3> {
        let x = x.clone() + self.attn.forward(self.attn_norm.forward(x), ctx);
        x.clone() + self.ffn.forward(self.ffn_norm.forward(x))
    }
}

/// Pre-norm transformer layer with a routed SwiGLU mixture.
#[derive(burn::module::Module, Debug)]
pub struct MoeBlock {
    pub attn_norm: RmsNorm,
    pub attn: Attention,
    pub ffn_norm: RmsNorm,
    pub moe: Mixture,
}

impl MoeBlock {
    pub(crate) fn new(config: &LoopMindConfig, device: &Device) -> Self {
        Self {
            attn_norm: rms_norm(config.d_model, config.rms_norm_eps, device),
            attn: Attention::new(config, device),
            ffn_norm: rms_norm(config.d_model, config.rms_norm_eps, device),
            moe: Mixture::new(config, device),
        }
    }

    pub(crate) fn forward(
        &self,
        x: Tensor<3>,
        force_routes: Option<Tensor<3, Int>>,
        ctx: &AttnContext,
        fused: &FusedExperts,
    ) -> (Tensor<3>, Tensor<1>, Tensor<1>) {
        let x = x.clone() + self.attn.forward(self.attn_norm.forward(x), ctx);
        let (ffn_out, aux, slots) =
            self.moe
                .forward_fused(self.ffn_norm.forward(x.clone()), force_routes, fused);
        (x + ffn_out, aux, slots)
    }
}
