//! Time one MoE layer, forward and backward, in Burn and in Candle.
//!
//! The layer is the default core layer: d_model 256, 32 SwiGLU experts of width 128, top-2.
//! Both sides run that algorithm on the same token count. Weights are initialized separately.
//!
//! ```text
//! cargo run -p moe-bakeoff --release
//! cargo run -p moe-bakeoff --release --features cuda
//! ```

use std::time::Instant;

use burn::module::Module;
use burn::tensor::{Distribution, Tensor};
use candle_core::{DType, Device as CandleDevice, Tensor as CandleTensor, Var};
use loopmind_model::{LoopMindConfig, Mixture};

const TOKENS: usize = 128;
const WARMUP: usize = 2;
const ITERS: usize = 5;

fn main() {
    let config = LoopMindConfig::default();
    println!(
        "moe layer  d_model={}  experts={}  width={}  top_k={}  tokens={}",
        config.d_model,
        config.n_experts,
        config.expert_intermediate_size,
        config.experts_per_token,
        TOKENS
    );
    let burn_ms = time_burn(&config);
    let candle_ms = time_candle(&config).expect("candle bakeoff");
    println!("burn_median_ms {burn_ms:.2}");
    println!("candle_median_ms {candle_ms:.2}");
    let faster = if burn_ms < candle_ms {
        "burn"
    } else {
        "candle"
    };
    let ratio = burn_ms.max(candle_ms) / burn_ms.min(candle_ms);
    println!("faster {faster}  ratio {ratio:.2}");
    println!("device {}", device_name());
}

fn device_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda:0"
    } else {
        "cpu (burn flex, candle cpu)"
    }
}

fn burn_device() -> burn::tensor::Device {
    #[cfg(feature = "cuda")]
    {
        burn::tensor::Device::cuda(0).autodiff()
    }
    #[cfg(not(feature = "cuda"))]
    {
        burn::tensor::Device::flex().autodiff()
    }
}

fn candle_device() -> CandleDevice {
    #[cfg(feature = "cuda")]
    {
        CandleDevice::new_cuda(0).expect("cuda device")
    }
    #[cfg(not(feature = "cuda"))]
    {
        CandleDevice::Cpu
    }
}

fn time_burn(config: &LoopMindConfig) -> f64 {
    let device = burn_device();
    device.seed(1);
    let moe = Mixture::new(config, &device).train();
    let input = Tensor::<3>::random(
        [1, TOKENS, config.d_model],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let mut samples = Vec::with_capacity(WARMUP + ITERS);
    for _ in 0..(WARMUP + ITERS) {
        let start = Instant::now();
        let (output, aux, _slots) = moe.forward(input.clone(), None);
        let loss = output.sum() + aux;
        let _grads = loss.backward();
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    median(&mut samples[WARMUP..])
}

fn time_candle(config: &LoopMindConfig) -> candle_core::Result<f64> {
    let device = candle_device();
    let layer = CandleMoe::new(config, &device)?;
    let input = CandleTensor::randn(0f32, 1.0, (1, TOKENS, config.d_model), &device)?;
    let mut samples = Vec::with_capacity(WARMUP + ITERS);
    for _ in 0..(WARMUP + ITERS) {
        let start = Instant::now();
        let (output, aux) = layer.forward(&input)?;
        let loss = (output.sum_all()? + aux)?;
        let grads = loss.backward()?;
        if grads.get(layer.router.as_tensor()).is_none() {
            candle_core::bail!("router gradient was not retained");
        }
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    Ok(median(&mut samples[WARMUP..]))
}

struct CandleMoe {
    router: Var,
    gates: Vec<Var>,
    ups: Vec<Var>,
    downs: Vec<Var>,
    top_k: usize,
}

impl CandleMoe {
    fn new(config: &LoopMindConfig, device: &CandleDevice) -> candle_core::Result<Self> {
        let d_model = config.d_model;
        let width = config.expert_intermediate_size;
        let mut gates = Vec::new();
        let mut ups = Vec::new();
        let mut downs = Vec::new();
        for _ in 0..config.n_experts {
            gates.push(var_randn((d_model, width), device)?);
            ups.push(var_randn((d_model, width), device)?);
            downs.push(var_randn((width, d_model), device)?);
        }
        Ok(Self {
            router: var_randn((d_model, config.n_experts), device)?,
            gates,
            ups,
            downs,
            top_k: config.experts_per_token,
        })
    }

    fn forward(&self, x: &CandleTensor) -> candle_core::Result<(CandleTensor, CandleTensor)> {
        let (batch, seq, d_model) = x.dims3()?;
        let flat = x.reshape((batch * seq, d_model))?;
        let logits = flat.matmul(self.router.as_tensor())?;
        let probs = softmax_last(&logits)?;
        let (sorted, order) = probs.sort_last_dim(false)?;
        let top_w = sorted.narrow(1, 0, self.top_k)?;
        let top_i = order.narrow(1, 0, self.top_k)?;
        let denom = top_w.sum_keepdim(1)?.clamp(TOPK_MIN, f64::MAX)?;
        let top_w = top_w.broadcast_div(&denom)?;
        let n_experts = self.gates.len();
        let mut mixed = CandleTensor::zeros(flat.shape(), flat.dtype(), flat.device())?;
        for expert_id in 0..n_experts {
            let column = expert_column(&top_i, &top_w, expert_id)?;
            let hidden = flat
                .matmul(self.gates[expert_id].as_tensor())?
                .silu()?
                .broadcast_mul(&flat.matmul(self.ups[expert_id].as_tensor())?)?;
            let expert = hidden.matmul(self.downs[expert_id].as_tensor())?;
            mixed = mixed.add(&expert.broadcast_mul(&column)?)?;
        }
        // A small router penalty so backward keeps the router. The Switch loss used by
        // LoopMind is a similar reduction; the expert SwiGLUs dominate the step.
        let mean_prob = probs.mean(0)?;
        let aux = mean_prob.sqr()?.sum_all()?.affine(n_experts as f64, 0.0)?;
        Ok((mixed.reshape((batch, seq, d_model))?, aux))
    }
}

fn softmax_last(x: &CandleTensor) -> candle_core::Result<CandleTensor> {
    let max = x.max_keepdim(1)?;
    let exp = x.broadcast_sub(&max)?.exp()?;
    exp.broadcast_div(&exp.sum_keepdim(1)?)
}

fn expert_column(
    indices: &CandleTensor,
    weights: &CandleTensor,
    expert_id: usize,
) -> candle_core::Result<CandleTensor> {
    let tokens = indices.dim(0)?;
    let device = indices.device();
    let id = CandleTensor::from_vec(vec![expert_id as u32; tokens], (tokens, 1), device)?;
    let mut column = CandleTensor::zeros((tokens, 1), DType::F32, device)?;
    for slot in 0..indices.dim(1)? {
        let chosen = indices.narrow(1, slot, 1)?;
        let weight = weights.narrow(1, slot, 1)?;
        let mask = chosen.eq(&id)?.to_dtype(DType::F32)?;
        column = column.add(&weight.broadcast_mul(&mask)?)?;
    }
    Ok(column)
}

const TOPK_MIN: f64 = 1.0e-9;

fn var_randn(shape: (usize, usize), device: &CandleDevice) -> candle_core::Result<Var> {
    Var::from_tensor(&CandleTensor::randn(0f32, 0.02, shape, device)?)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}
