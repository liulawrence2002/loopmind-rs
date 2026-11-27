//! One pretrain step: sampled loops, shifted cross-entropy, aux loss, AdamW.
//!
//! Weight decay is not applied to embeddings or RMSNorm scales. Gradients are
//! clipped by their global L2 norm. Weights are f32 unless the run asks for
//! [`Precision::Auto`]. Then a CUDA build tries bf16, then f16, and stays on
//! f32 if that matmul does not return finite values. The stored weights use
//! that dtype and Burn updates them in place, so there is no f32 master copy.
//! An AdamW step near `lr` is below half a bf16 ulp of an RMSNorm gain of 1.0,
//! which is why f32 is the default.

use std::path::Path;
use std::sync::Mutex;

/// Burn's flex RNG is process-global. Seed and initialization must not interleave
/// with another run, or two engines with the same seed diverge.
static INIT_RNG: Mutex<()> = Mutex::new(());

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams};
use burn::tensor::{DType, Device, Tensor};
use loopmind_data::{PretrainBatch, SftBatch};

use crate::batches::{pretrain_batch, sft_batch};
use crate::config::LoopMindConfig;
use crate::loopmind::{greedy_continue, sample_continue, GenerateOptions, LoopMind};
use crate::loss::{masked_shifted_nll_sum, shifted_token_cross_entropy};
use crate::rng::TrainRng;

/// Where the run executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineDevice {
    Cpu,
    Cuda,
    Wgpu,
}

impl EngineDevice {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "wgpu" => Ok(Self::Wgpu),
            other => Err(format!("unknown device {other}; use cpu, cuda, or wgpu")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
        }
    }
}

/// Weight dtype a run asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// f32 weights and optimizer state on every device.
    F32,
    /// On CUDA, probe bf16 then f16 and cast the weights to the first that works.
    Auto,
}

impl Precision {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "f32" => Ok(Self::F32),
            "auto" => Ok(Self::Auto),
            other => Err(format!("unknown precision {other}; use f32 or auto")),
        }
    }
}

/// AdamW settings that are not the learning rate. The schedule lives with the run.
#[derive(Debug, Clone, Copy)]
pub struct OptimSettings {
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub grad_clip: f32,
}

/// Result of one microbatch backward.
#[derive(Debug, Clone)]
pub struct MicroStats {
    pub loss: f32,
    pub aux: f32,
    pub loops: Vec<usize>,
    pub expert_slots: Vec<Vec<f32>>,
    pub tokens: u64,
}

/// Result of one SFT microbatch backward.
#[derive(Debug, Clone)]
pub struct SftStats {
    /// Sum of next-token NLL over supervised targets.
    pub nll_sum: f32,
    /// Supervised targets in the row (mask 1 at positions `1..seq`).
    pub targets: u64,
    pub aux: f32,
    pub loops: Vec<usize>,
    pub expert_slots: Vec<Vec<f32>>,
    /// Padded tokens fed to the model.
    pub tokens: u64,
}

/// Result of the optimizer step that consumes accumulated gradients.
#[derive(Debug, Clone, Copy)]
pub struct ApplyStats {
    pub grad_norm: f32,
}

pub struct PretrainEngine {
    model: Option<LoopMind>,
    optim: burn::optim::ModuleOptimizer,
    accumulator: GradientsAccumulator<LoopMind>,
    pending: u32,
    rng: TrainRng,
    device: Device,
    precision: String,
    kind: EngineDevice,
    grad_clip: f32,
}

impl PretrainEngine {
    pub fn new(
        config: &LoopMindConfig,
        seed: u64,
        kind: EngineDevice,
        settings: OptimSettings,
        requested: Precision,
    ) -> Result<Self, String> {
        let device = burn_device(kind)?;
        let _guard = INIT_RNG.lock().expect("init rng lock");
        device.seed(seed);
        let mut model = LoopMind::init(config, &device);
        drop(_guard);
        let precision = match requested {
            Precision::F32 => "f32",
            Precision::Auto => resolve_precision(kind, &device),
        }
        .to_string();
        if let Some(dtype) = dtype_of(&precision) {
            model = cast_parameters(model, dtype);
        }
        Ok(Self {
            optim: adamw(settings),
            accumulator: GradientsAccumulator::new(),
            pending: 0,
            rng: TrainRng::new(seed ^ 0xD1CE_5EED),
            model: Some(model),
            device,
            precision,
            kind,
            grad_clip: settings.grad_clip,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load(
        config: &LoopMindConfig,
        seed: u64,
        kind: EngineDevice,
        settings: OptimSettings,
        model_path: &Path,
        optim_path: &Path,
        rng_state: u64,
        precision: String,
    ) -> Result<Self, String> {
        let device = burn_device(kind)?;
        let _guard = INIT_RNG.lock().expect("init rng lock");
        device.seed(seed);
        let model = LoopMind::init(config, &device)
            .try_load_file(model_path)
            .map_err(|error| error.to_string())?;
        drop(_guard);
        let optim = adamw(settings)
            .load(optim_path)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            model: Some(model),
            optim,
            accumulator: GradientsAccumulator::new(),
            pending: 0,
            rng: TrainRng::from_state(rng_state),
            device,
            precision,
            kind,
            grad_clip: settings.grad_clip,
        })
    }

    /// Weights from `model_path` with a fresh AdamW. SFT starts from a pretrain
    /// checkpoint this way, and inference uses it because it never reads `optim.bpk`.
    pub fn from_weights(
        config: &LoopMindConfig,
        seed: u64,
        kind: EngineDevice,
        settings: OptimSettings,
        model_path: &Path,
        requested: Precision,
    ) -> Result<Self, String> {
        let device = burn_device(kind)?;
        let _guard = INIT_RNG.lock().expect("init rng lock");
        device.seed(seed);
        let mut model = LoopMind::init(config, &device)
            .try_load_file(model_path)
            .map_err(|error| error.to_string())?;
        drop(_guard);
        let precision = match requested {
            Precision::F32 => "f32",
            Precision::Auto => resolve_precision(kind, &device),
        }
        .to_string();
        if let Some(dtype) = dtype_of(&precision) {
            model = cast_parameters(model, dtype);
        }
        Ok(Self {
            model: Some(model),
            optim: adamw(settings),
            accumulator: GradientsAccumulator::new(),
            pending: 0,
            rng: TrainRng::new(seed ^ 0xD1CE_5EED),
            device,
            precision,
            kind,
            grad_clip: settings.grad_clip,
        })
    }

    pub fn precision(&self) -> &str {
        &self.precision
    }

    pub fn device_name(&self) -> &'static str {
        self.kind.name()
    }

    pub fn config(&self) -> &LoopMindConfig {
        &self.net().config
    }

    fn net(&self) -> &LoopMind {
        self.model.as_ref().expect("model is loaded")
    }

    pub fn rng_state(&self) -> u64 {
        self.rng.state()
    }

    pub fn set_rng_state(&mut self, state: u64) {
        self.rng = TrainRng::from_state(state);
    }

    pub fn accumulate(&mut self, batch: &PretrainBatch) -> Result<MicroStats, String> {
        if batch.seq_len < 2 {
            return Err("a packed sequence needs at least 2 tokens".into());
        }
        let tokens = pretrain_batch(batch, &self.device);
        let mut rng = self.rng.clone();
        let model = self.model.as_ref().expect("model is loaded");
        let out = model.forward_sampled(tokens.clone(), &mut rng);
        self.rng = rng;
        let aux = out.aux_loss.clone();
        let loss = shifted_token_cross_entropy(out.logits, tokens) + aux.clone();
        let [loss_value, aux_value] = read_pair(&loss, &aux);
        if !loss_value.is_finite() {
            return Err("pretrain loss is not finite".into());
        }
        self.backward(loss);
        Ok(MicroStats {
            loss: loss_value,
            aux: aux_value,
            loops: out.loops,
            expert_slots: out.expert_slots,
            tokens: (batch.batch * batch.seq_len) as u64,
        })
    }

    /// One SFT microbatch. The backward loss is `token_scale * masked_nll_sum + aux_scale * aux`.
    ///
    /// The driver sets `token_scale` to one over the step's supervised targets and
    /// `aux_scale` to one over its rows, then calls [`PretrainEngine::apply_summed`], so a
    /// step is the token-mean loss over every supervised target in it.
    pub fn accumulate_sft(
        &mut self,
        batch: &SftBatch,
        token_scale: f32,
        aux_scale: f32,
    ) -> Result<SftStats, String> {
        if batch.seq_len < 2 {
            return Err("an SFT row needs at least 2 tokens".into());
        }
        let targets = batch.supervised_targets();
        let (tokens, mask) = sft_batch(batch, &self.device);
        let mut rng = self.rng.clone();
        let model = self.model.as_ref().expect("model is loaded");
        let out = model.forward_sampled(tokens.clone(), &mut rng);
        self.rng = rng;
        let aux = out.aux_loss.clone();
        let nll = masked_shifted_nll_sum(out.logits, tokens, mask);
        let [nll_value, aux_value] = read_pair(&nll, &aux);
        if !nll_value.is_finite() {
            return Err("sft loss is not finite".into());
        }
        self.backward(nll.mul_scalar(token_scale) + aux.mul_scalar(aux_scale));
        Ok(SftStats {
            nll_sum: nll_value,
            targets,
            aux: aux_value,
            loops: out.loops,
            expert_slots: out.expert_slots,
            tokens: (batch.batch * batch.seq_len) as u64,
        })
    }

    fn backward(&mut self, loss: Tensor<1>) {
        let mut grads = GradientsParams::from_grads(loss.backward(), self.net());
        detach_grads(self.net(), &mut grads);
        let model = self.model.take().expect("model is loaded");
        self.accumulator.accumulate(&model, grads);
        self.model = Some(model);
        self.pending += 1;
    }

    /// Average the accumulated microbatch gradients, clip, and step.
    pub fn apply(&mut self, learning_rate: f64) -> Result<ApplyStats, String> {
        if self.pending == 0 {
            return Err("no gradients to apply".into());
        }
        self.apply_scaled(learning_rate, 1.0 / self.pending as f32)
    }

    /// Step on the summed gradients. Microbatch losses were already scaled by the caller.
    pub fn apply_summed(&mut self, learning_rate: f64) -> Result<ApplyStats, String> {
        if self.pending == 0 {
            return Err("no gradients to apply".into());
        }
        self.apply_scaled(learning_rate, 1.0)
    }

    fn apply_scaled(&mut self, learning_rate: f64, scale: f32) -> Result<ApplyStats, String> {
        let mut grads = self.accumulator.grads();
        // The norm of `scale * g` is `scale * |g|`, so averaging and clipping share one rescale.
        let grad_norm = grad_norm(self.net(), &grads) * scale;
        let clip = self.grad_clip;
        let factor = if clip > 0.0 && grad_norm > clip {
            scale * clip / grad_norm
        } else {
            scale
        };
        if factor != 1.0 {
            scale_grads(self.net(), &mut grads, factor);
        }
        let model = self.model.take().expect("model is loaded");
        self.model = Some(self.optim.step(learning_rate, model, grads));
        self.pending = 0;
        Ok(ApplyStats { grad_norm })
    }

    /// Loss at a fixed `r`, without an autodiff graph. An untruncated forward on the
    /// training device would keep every pass's activations alive, twice the four
    /// backprop passes of a training step at `r = 8`.
    pub fn validate(&self, batch: &PretrainBatch, loops: usize) -> Result<f32, String> {
        let tokens = pretrain_batch(batch, &self.plain_device());
        let out = self.net().valid().forward(tokens.clone(), loops);
        let loss = shifted_token_cross_entropy(out.logits, tokens);
        Ok(loss.into_scalar::<f32>())
    }

    /// Masked NLL sum and supervised-target count at a fixed `r`, without an autodiff graph.
    pub fn validate_sft(&self, batch: &SftBatch, loops: usize) -> Result<(f32, u64), String> {
        let (tokens, mask) = sft_batch(batch, &self.plain_device());
        let out = self.net().valid().forward(tokens.clone(), loops);
        let nll = masked_shifted_nll_sum(out.logits, tokens, mask);
        Ok((nll.into_scalar::<f32>(), batch.supervised_targets()))
    }

    pub fn greedy(&self, ids: Vec<u32>, loops: usize, max_new: usize, eos: u32) -> Vec<u32> {
        let device = self.plain_device();
        greedy_continue(&self.net().valid(), ids, loops, max_new, eos, &device)
    }

    /// Sampled continuation that stops on any id in `stop`. Returns only the new ids.
    pub fn generate(&self, ids: Vec<u32>, options: &GenerateOptions, stop: &[u32]) -> Vec<u32> {
        let device = self.plain_device();
        sample_continue(&self.net().valid(), ids, options, stop, &device)
    }

    fn plain_device(&self) -> Device {
        self.device.clone().inner()
    }

    pub fn save(&self, model_path: &Path, optim_path: &Path) -> Result<(), String> {
        if let Some(parent) = model_path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        self.net()
            .clone()
            .save_file(model_path)
            .map_err(|error| error.to_string())?;
        self.optim
            .save(optim_path)
            .map_err(|error| error.to_string())?;
        Ok(())
    }
}

fn adamw(settings: OptimSettings) -> burn::optim::ModuleOptimizer {
    let decay = AdamWConfig::new()
        .with_beta_1(settings.beta1)
        .with_beta_2(settings.beta2)
        .with_epsilon(settings.epsilon)
        .with_weight_decay(settings.weight_decay)
        .init();
    let frozen_decay = AdamWConfig::new()
        .with_beta_1(settings.beta1)
        .with_beta_2(settings.beta2)
        .with_epsilon(settings.epsilon)
        .with_weight_decay(0.0)
        .build();
    let group = burn::module::ParamGroup::from_any_predicates(vec![
        "embed".to_string(),
        "norm".to_string(),
    ]);
    decay.with_group(group, frozen_decay, None)
}

fn burn_device(kind: EngineDevice) -> Result<Device, String> {
    match kind {
        EngineDevice::Cpu => Ok(Device::flex().autodiff()),
        EngineDevice::Cuda => cuda_device(),
        EngineDevice::Wgpu => wgpu_device(),
    }
}

#[cfg(feature = "cuda")]
fn cuda_device() -> Result<Device, String> {
    Ok(Device::cuda(0).autodiff())
}

#[cfg(not(feature = "cuda"))]
fn cuda_device() -> Result<Device, String> {
    Err("this binary has no CUDA backend; rebuild with --features cuda".into())
}

#[cfg(feature = "wgpu")]
fn wgpu_device() -> Result<Device, String> {
    Ok(Device::wgpu(burn::tensor::DeviceKind::DefaultDevice).autodiff())
}

#[cfg(not(feature = "wgpu"))]
fn wgpu_device() -> Result<Device, String> {
    Err("this binary has no wgpu backend; rebuild with --features wgpu".into())
}

fn resolve_precision(kind: EngineDevice, device: &Device) -> &'static str {
    if kind != EngineDevice::Cuda {
        return "f32";
    }
    if probe(device, DType::BF16) {
        "bf16"
    } else if probe(device, DType::F16) {
        "f16"
    } else {
        "f32"
    }
}

fn dtype_of(precision: &str) -> Option<DType> {
    match precision {
        "bf16" => Some(DType::BF16),
        "f16" => Some(DType::F16),
        _ => None,
    }
}

fn probe(device: &Device, dtype: DType) -> bool {
    let left = Tensor::<2>::ones([8, 8], device).cast(dtype);
    let value = left.clone().matmul(left).cast(DType::F32);
    value
        .into_data()
        .try_into_vec::<f32>()
        .map(|items| items.iter().all(|item| item.is_finite()))
        .unwrap_or(false)
}

fn cast_parameters(model: LoopMind, dtype: DType) -> LoopMind {
    struct Cast(DType);
    impl ModuleMapper for Cast {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let updated = param.val().cast(self.0).detach().require_grad();
            param.map(|_| updated)
        }
    }
    model.map(&mut Cast(dtype))
}

/// Global L2 norm of the gradients, read back in one device sync.
fn grad_norm(model: &LoopMind, grads: &GradientsParams) -> f32 {
    struct Norm<'a> {
        grads: &'a GradientsParams,
        squares: Vec<Tensor<1>>,
    }
    impl ModuleVisitor for Norm<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let Some(grad) = self.grads.get::<D>(param.id) else {
                return;
            };
            self.squares
                .push(grad.cast(DType::F32).powf_scalar(2.0).sum());
        }
    }
    let mut norm = Norm {
        grads,
        squares: Vec::new(),
    };
    model.visit(&mut norm);
    if norm.squares.is_empty() {
        return 0.0;
    }
    Tensor::cat(norm.squares, 0)
        .sum()
        .into_scalar::<f32>()
        .sqrt()
}

/// Two scalars in one device read.
fn read_pair(first: &Tensor<1>, second: &Tensor<1>) -> [f32; 2] {
    let values = Tensor::cat(
        vec![
            first.clone().detach().cast(DType::F32),
            second.clone().detach().cast(DType::F32),
        ],
        0,
    )
    .into_data()
    .try_into_vec::<f32>()
    .expect("loss scalars are f32");
    [values[0], values[1]]
}

fn detach_grads(model: &LoopMind, grads: &mut GradientsParams) {
    struct Detach<'a> {
        grads: &'a mut GradientsParams,
    }
    impl ModuleVisitor for Detach<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let Some(grad) = self.grads.remove::<D>(param.id) else {
                return;
            };
            self.grads.register(param.id, grad.detach());
        }
    }
    let mut detach = Detach { grads };
    model.visit(&mut detach);
}

fn scale_grads(model: &LoopMind, grads: &mut GradientsParams, scale: f32) {
    struct Scale<'a> {
        grads: &'a mut GradientsParams,
        scale: f32,
    }
    impl ModuleVisitor for Scale<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let Some(grad) = self.grads.remove::<D>(param.id) else {
                return;
            };
            self.grads.register(param.id, grad.mul_scalar(self.scale));
        }
    }
    let mut scale = Scale { grads, scale };
    model.visit(&mut scale);
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, TensorData};

    /// Every f32 helper tensor (RoPE tables, expert one-hot, slot fractions) must be
    /// cast to the weights' dtype. On flex a mix panics; on CUDA it read as garbage
    /// and the first bf16 loss was not finite.
    #[test]
    fn bf16_weights_give_a_finite_loss_and_gradient() {
        let device = Device::flex().autodiff();
        device.seed(42);
        let config = LoopMindConfig::tiny();
        let model = cast_parameters(LoopMind::init(&config, &device), DType::BF16);
        let ids: Vec<i32> = (0..12).map(|index| index * 37 % 128).collect();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, 12]), &device);
        let out = model.forward_train_at(tokens.clone(), 3, None);
        assert_eq!(out.logits.dtype(), DType::BF16);
        let loss = shifted_token_cross_entropy(out.logits, tokens) + out.aux_loss;
        let value = loss.clone().cast(DType::F32).into_scalar::<f32>();
        assert!(value.is_finite(), "bf16 loss {value}");
        let grads = loss.backward();
        let grad = model
            .embed
            .weight
            .val()
            .grad(&grads)
            .expect("embedding gradient")
            .cast(DType::F32)
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
        assert!(grad.iter().all(|entry| entry.is_finite()));
    }
}
