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
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{DType, Device, Tensor};
use loopmind_data::{PretrainBatch, SftBatch};

use crate::batches::{pretrain_batch, sft_batch};
use crate::config::LoopMindConfig;
use crate::loopmind::{
    greedy_continue, sample_batch, sample_continue, sample_stream, GenerateOptions, LoopMind,
};
use crate::loss::{
    masked_shifted_greedy_misses_rows, masked_shifted_logprob_rows, masked_shifted_nll_sum,
    shifted_token_cross_entropy,
};
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

/// Result of one DPO pair backward. Rewards are `beta * (log pi - log ref)` of a response.
#[derive(Debug, Clone)]
pub struct DpoStats {
    /// `-log sigmoid(margin)`.
    pub loss: f32,
    /// `chosen_reward - rejected_reward`.
    pub margin: f32,
    pub chosen_reward: f32,
    pub rejected_reward: f32,
    /// Token-mean NLL of the chosen response under the policy.
    pub chosen_nll: f32,
    pub aux: f32,
    pub loops: usize,
    pub tokens: u64,
}

/// Result of the optimizer step that consumes accumulated gradients.
#[derive(Debug, Clone, Copy)]
pub struct ApplyStats {
    pub grad_norm: f32,
}

pub struct PretrainEngine {
    model: Option<LoopMind>,
    /// Frozen DPO reference on the plain device. `None` outside DPO.
    reference: Option<LoopMind>,
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
            reference: None,
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
            reference: None,
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
            reference: None,
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

    /// Per-row masked NLL sums at a fixed `r`, without an autodiff graph. Rows share a width.
    pub fn nll_rows(&self, batch: &SftBatch, loops: usize) -> Result<Vec<f32>, String> {
        let (tokens, mask) = sft_batch(batch, &self.plain_device());
        let out = self.net().valid().forward(tokens.clone(), loops);
        masked_shifted_logprob_rows(out.logits, tokens, mask)
            .neg()
            .cast(DType::F32)
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|error| format!("{error:?}"))
    }

    /// Per-row masked log-probability sum, and whether greedy decoding reproduces every
    /// supervised target, at a fixed `r` without an autodiff graph. Rows share a width.
    pub fn score_rows(&self, batch: &SftBatch, loops: usize) -> Result<Vec<(f32, bool)>, String> {
        let (tokens, mask) = sft_batch(batch, &self.plain_device());
        let out = self.net().valid().forward(tokens.clone(), loops);
        let misses =
            masked_shifted_greedy_misses_rows(out.logits.clone(), tokens.clone(), mask.clone())
                .into_data()
                .convert::<i64>()
                .try_into_vec::<i64>()
                .map_err(|error| format!("{error:?}"))?;
        let logprobs = masked_shifted_logprob_rows(out.logits, tokens, mask)
            .cast(DType::F32)
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|error| format!("{error:?}"))?;
        Ok(logprobs
            .into_iter()
            .zip(misses)
            .map(|(logprob, missed)| (logprob, missed == 0))
            .collect())
    }

    /// Freeze the weights at `model_path` as the DPO reference, on the plain device.
    pub fn load_reference(&mut self, model_path: &Path) -> Result<(), String> {
        let device = self.plain_device();
        let mut reference = LoopMind::init(self.config(), &device)
            .try_load_file(model_path)
            .map_err(|error| error.to_string())?;
        if let Some(dtype) = dtype_of(&self.precision) {
            reference = cast_parameters(reference, dtype);
        }
        self.reference = Some(reference.valid());
        Ok(())
    }

    /// One DPO pair. Row 0 of `pair` is the chosen response, row 1 the rejected one.
    ///
    /// Both rows and the reference share one sampled loop count, so `r` cancels out of
    /// the log-ratio. The backward loss is
    /// `scale * (dpo + nll_weight * chosen token-mean NLL) + aux_scale * aux`;
    /// call [`PretrainEngine::apply_summed`] after the step's pairs.
    pub fn accumulate_dpo(
        &mut self,
        pair: &SftBatch,
        beta: f32,
        nll_weight: f32,
        scale: f32,
        aux_scale: f32,
    ) -> Result<DpoStats, String> {
        if pair.batch != 2 || pair.seq_len < 2 {
            return Err("a DPO pair is two rows of at least 2 tokens".into());
        }
        let targets = pair.targets_per_row();
        let config = self.config().clone();
        let mut rng = self.rng.clone();
        let loops = rng.sample_loops(config.loop_count_mean, config.loop_count_max);
        self.rng = rng;
        let reference_lp = self.reference_logprobs(pair, loops)?;
        let (tokens, mask) = sft_batch(pair, &self.device);
        let model = self.model.as_ref().expect("model is loaded");
        let out = model.forward_train_at(tokens.clone(), loops, None);
        let policy = masked_shifted_logprob_rows(out.logits, tokens, mask);
        let reference_lp = if self.device.is_autodiff() {
            reference_lp.autodiff()
        } else {
            reference_lp
        };
        let ratio = policy.clone() - reference_lp;
        let chosen = ratio.clone().narrow(0, 0, 1).mul_scalar(beta);
        let rejected = ratio.narrow(0, 1, 1).mul_scalar(beta);
        let margin = chosen.clone() - rejected.clone();
        let dpo = log_sigmoid(margin.clone()).neg();
        let chosen_nll = policy
            .narrow(0, 0, 1)
            .neg()
            .div_scalar(targets[0].max(1) as f32);
        let aux = out.aux_loss;
        let values = read_values(&[&dpo, &margin, &chosen, &rejected, &chosen_nll, &aux]);
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("dpo values are not finite: {values:?}"));
        }
        let loss =
            (dpo + chosen_nll.mul_scalar(nll_weight)).mul_scalar(scale) + aux.mul_scalar(aux_scale);
        self.backward(loss);
        Ok(DpoStats {
            loss: values[0],
            margin: values[1],
            chosen_reward: values[2],
            rejected_reward: values[3],
            chosen_nll: values[4],
            aux: values[5],
            loops,
            tokens: (pair.batch * pair.seq_len) as u64,
        })
    }

    /// DPO loss and margin of one pair at a fixed `r`, without an autodiff graph.
    pub fn validate_dpo(
        &self,
        pair: &SftBatch,
        beta: f32,
        loops: usize,
    ) -> Result<(f32, f32), String> {
        let reference_lp = self.reference_logprobs(pair, loops)?;
        let (tokens, mask) = sft_batch(pair, &self.plain_device());
        let out = self.net().valid().forward(tokens.clone(), loops);
        let ratio = masked_shifted_logprob_rows(out.logits, tokens, mask) - reference_lp;
        let margin = (ratio.clone().narrow(0, 0, 1) - ratio.narrow(0, 1, 1)).mul_scalar(beta);
        let dpo = log_sigmoid(margin.clone()).neg();
        let values = read_values(&[&dpo, &margin]);
        Ok((values[0], values[1]))
    }

    fn reference_logprobs(&self, pair: &SftBatch, loops: usize) -> Result<Tensor<1>, String> {
        let reference = self
            .reference
            .as_ref()
            .ok_or("load_reference before a DPO step")?;
        let (tokens, mask) = sft_batch(pair, &self.plain_device());
        let logits = reference.forward(tokens.clone(), loops).logits;
        Ok(masked_shifted_logprob_rows(logits, tokens, mask))
    }

    /// Sampled continuations of equal-length prompts in one batch. See [`sample_batch`].
    pub fn generate_batch(
        &self,
        prompts: &[Vec<u32>],
        options: &GenerateOptions,
        stop: &[u32],
    ) -> Vec<Vec<u32>> {
        let device = self.plain_device();
        sample_batch(&self.net().valid(), prompts, options, stop, &device)
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

    /// [`Self::generate`] that hands each new id to `on_token` as it is picked, and
    /// stops early when `on_token` returns false. See [`sample_stream`].
    pub fn generate_stream(
        &self,
        ids: Vec<u32>,
        options: &GenerateOptions,
        stop: &[u32],
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Vec<u32> {
        let device = self.plain_device();
        sample_stream(&self.net().valid(), ids, options, stop, &device, on_token)
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

/// Scalars in one device read.
fn read_values(values: &[&Tensor<1>]) -> Vec<f32> {
    Tensor::cat(
        values
            .iter()
            .map(|value| (*value).clone().detach().inner().cast(DType::F32))
            .collect(),
        0,
    )
    .into_data()
    .try_into_vec::<f32>()
    .expect("scalars are f32")
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

    fn settings() -> OptimSettings {
        OptimSettings {
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1.0e-8,
            grad_clip: 1.0,
        }
    }

    #[test]
    fn batched_greedy_matches_one_prompt_at_a_time() {
        let engine = PretrainEngine::new(
            &LoopMindConfig::tiny(),
            3,
            EngineDevice::Cpu,
            settings(),
            Precision::F32,
        )
        .unwrap();
        let options = GenerateOptions {
            loops: 3,
            max_new: 6,
            temperature: 0.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: 0,
        };
        let prompts = vec![vec![1, 5, 9, 2], vec![7, 3, 3, 8]];
        let batched = engine.generate_batch(&prompts, &options, &[]);
        for (prompt, row) in prompts.iter().zip(&batched) {
            assert_eq!(&engine.generate(prompt.clone(), &options, &[]), row);
            assert_eq!(row.len(), 6);
        }
    }

    #[test]
    fn streaming_reports_each_id_and_stops_when_asked() {
        let engine = PretrainEngine::new(
            &LoopMindConfig::tiny(),
            3,
            EngineDevice::Cpu,
            settings(),
            Precision::F32,
        )
        .unwrap();
        let options = GenerateOptions {
            loops: 2,
            max_new: 6,
            temperature: 0.8,
            top_k: 5,
            repetition_penalty: 1.1,
            seed: 7,
        };
        let prompt = vec![1, 5, 9, 2];
        let whole = engine.generate(prompt.clone(), &options, &[]);
        let mut seen = Vec::new();
        let streamed = engine.generate_stream(prompt.clone(), &options, &[], &mut |id| {
            seen.push(id);
            true
        });
        assert_eq!(streamed, whole);
        assert_eq!(seen, whole);

        let mut count = 0;
        let cut = engine.generate_stream(prompt, &options, &[], &mut |_| {
            count += 1;
            count < 2
        });
        assert_eq!(cut, whole[..2]);
    }

    #[test]
    fn dpo_starts_at_ln2_and_widens_the_margin() {
        use loopmind_data::SftRow;
        let config = LoopMindConfig::tiny();
        let mut engine =
            PretrainEngine::new(&config, 4, EngineDevice::Cpu, settings(), Precision::F32).unwrap();
        let dir = std::env::temp_dir().join(format!("loopmind-dpo-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        engine
            .save(&dir.join("model.bpk"), &dir.join("optim.bpk"))
            .unwrap();
        engine.load_reference(&dir.join("model.bpk")).unwrap();
        let pair = SftBatch::from_rows(
            &[
                SftRow {
                    tokens: vec![0, 3, 10, 11, 40, 41, 4, 1],
                    mask: vec![0, 0, 0, 0, 1, 1, 1, 1],
                },
                SftRow {
                    tokens: vec![0, 3, 10, 11, 60, 61, 62],
                    mask: vec![0, 0, 0, 0, 1, 1, 1],
                },
            ],
            2,
            4,
        );
        let first = engine.accumulate_dpo(&pair, 0.5, 0.0, 1.0, 0.0).unwrap();
        assert!(
            (first.loss - std::f32::consts::LN_2).abs() < 1e-4,
            "{first:?}"
        );
        assert!(first.margin.abs() < 1e-4, "{first:?}");
        engine.apply_summed(1.0e-2).unwrap();
        let mut last = first;
        for _ in 0..15 {
            last = engine.accumulate_dpo(&pair, 0.5, 0.0, 1.0, 0.0).unwrap();
            engine.apply_summed(1.0e-2).unwrap();
        }
        assert!(last.margin > 0.5, "{last:?}");
        let (val_loss, val_margin) = engine.validate_dpo(&pair, 0.5, 2).unwrap();
        assert!(val_loss < std::f32::consts::LN_2 && val_margin > 0.0);
    }

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
