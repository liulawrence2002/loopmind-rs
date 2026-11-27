//! Huginn-style stack: embedding, dense prelude, looped MoE core, dense coda, tied head.
//!
//! ```text
//! e = Prelude(Embed(tokens))
//! s = e
//! for pass in 0..r:
//!     s = Core(s) + e          # prelude state is added again every pass
//! logits = FinalRMSNorm(Coda(s)) @ Eᵀ
//! ```
//!
//! `Core` is the two shared MoE layers. Each layer is pre-norm attention, then pre-norm SwiGLU.
//! Training samples `r` per sequence from a clamped Poisson and detaches passes that sit outside
//! the last `bptt_passes` iterations. Sampled batches run one sequence at a time so each row has
//! its own loop count.

use std::path::Path;

use burn::module::{Module, ModuleMapper, ModuleVisitor};
use burn::nn::{Embedding, EmbeddingConfig, Initializer, RmsNorm};
use burn::tensor::{Int, Tensor, TensorData};
use serde::Serialize;

use crate::blocks::{rms_norm, DenseBlock, MoeBlock};
use crate::config::LoopMindConfig;
use crate::rng::TrainRng;

/// Logits and the scaled load-balancing loss for one forward.
#[derive(Debug)]
pub struct ForwardOutput {
    /// `[batch, seq, vocab]`.
    pub logits: Tensor<3>,
    /// Scalar load-balancing loss already multiplied by [`LoopMindConfig::aux_loss_coef`].
    pub aux_loss: Tensor<1>,
    /// Loop count used for each batch row.
    pub loops: Vec<usize>,
    /// Mean fraction of top-k slots per expert, one vector per core layer, averaged over loop passes.
    pub expert_slots: Vec<Vec<f32>>,
}

/// The LoopMind network. Burn types stay in this crate's API.
#[derive(Module, Debug)]
pub struct LoopMind {
    pub embed: Embedding,
    pub prelude: Vec<DenseBlock>,
    pub core: Vec<MoeBlock>,
    pub coda: Vec<DenseBlock>,
    pub final_norm: RmsNorm,
    #[module(skip)]
    pub config: LoopMindConfig,
}

impl LoopMind {
    /// Build the network on `device`. Parameters require gradients when `device` has autodiff.
    pub fn init(config: &LoopMindConfig, device: &burn::tensor::Device) -> Self {
        config.validate().expect("LoopMindConfig failed validation");
        let embed = EmbeddingConfig::new(config.vocab_size, config.d_model)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);
        let prelude = (0..config.n_prelude_layers)
            .map(|_| DenseBlock::new(config, device))
            .collect();
        let core = (0..config.n_core_layers)
            .map(|_| MoeBlock::new(config, device))
            .collect();
        let coda = (0..config.n_coda_layers)
            .map(|_| DenseBlock::new(config, device))
            .collect();
        let final_norm = rms_norm(config.d_model, config.rms_norm_eps, device);
        Self {
            embed,
            prelude,
            core,
            coda,
            final_norm,
            config: config.clone(),
        }
    }

    /// Inference forward. `loops` is the dial `r` and is not truncated.
    pub fn forward(&self, tokens: Tensor<2, Int>, loops: usize) -> ForwardOutput {
        self.forward_opts(tokens, loops, false, None)
    }

    /// Training forward at a fixed `r`, detaching passes outside the last `bptt_passes`.
    pub fn forward_train_at(
        &self,
        tokens: Tensor<2, Int>,
        loops: usize,
        force_routes: Option<Tensor<3, Int>>,
    ) -> ForwardOutput {
        self.forward_opts(tokens, loops, true, force_routes)
    }

    /// Sample one loop count per sequence, then run truncated backprop through the tail.
    pub fn forward_sampled(&self, tokens: Tensor<2, Int>, rng: &mut TrainRng) -> ForwardOutput {
        let batch = tokens.dims()[0];
        let mut logits = Vec::with_capacity(batch);
        let mut aux_sum: Option<Tensor<1>> = None;
        let mut loops = Vec::with_capacity(batch);
        let mut slot_sum: Vec<Vec<f32>> = Vec::new();
        for index in 0..batch {
            let row = tokens.clone().narrow(0, index, 1);
            let r = rng.sample_loops(self.config.loop_count_mean, self.config.loop_count_max);
            let out = self.forward_opts(row, r, true, None);
            logits.push(out.logits);
            aux_sum = Some(match aux_sum {
                Some(sum) => sum + out.aux_loss,
                None => out.aux_loss,
            });
            if slot_sum.is_empty() {
                slot_sum = out.expert_slots;
            } else {
                add_slots(&mut slot_sum, &out.expert_slots);
            }
            loops.push(r);
        }
        scale_slots(&mut slot_sum, 1.0 / batch as f32);
        let aux_loss = aux_sum
            .unwrap_or_else(|| Tensor::<1>::zeros([1], &tokens.device()))
            .mul_scalar(1.0 / batch as f32);
        ForwardOutput {
            logits: Tensor::cat(logits, 0),
            aux_loss,
            loops,
            expert_slots: slot_sum,
        }
    }

    fn forward_opts(
        &self,
        tokens: Tensor<2, Int>,
        loops: usize,
        truncate: bool,
        force_routes: Option<Tensor<3, Int>>,
    ) -> ForwardOutput {
        assert!(loops >= 1, "loop count must be at least 1");
        let bptt = if truncate {
            self.config.bptt_passes
        } else {
            usize::MAX
        };
        let embedded = self.embed.forward(tokens);
        let e = dense_stack(&self.prelude, embedded);
        let (state, aux, expert_slots) = self.recur(e, loops, bptt, force_routes);
        let hidden = self.final_norm.forward(dense_stack(&self.coda, state));
        let logits = tied_logits(hidden, &self.embed);
        let aux_loss = aux.mul_scalar(self.config.aux_loss_coef);
        ForwardOutput {
            logits,
            aux_loss,
            loops: vec![loops],
            expert_slots,
        }
    }

    fn recur(
        &self,
        e: Tensor<3>,
        loops: usize,
        bptt: usize,
        force_routes: Option<Tensor<3, Int>>,
    ) -> (Tensor<3>, Tensor<1>, Vec<Vec<f32>>) {
        let mut state = e.clone();
        let mut aux: Option<Tensor<1>> = None;
        let mut slot_sum: Vec<Vec<f32>> = Vec::new();
        for pass in 0..loops {
            let (core_out, pass_aux, pass_slots) = self.apply_core(state, force_routes.clone());
            if slot_sum.is_empty() {
                slot_sum = pass_slots;
            } else {
                add_slots(&mut slot_sum, &pass_slots);
            }
            let updated = core_out + e.clone();
            let early = pass.saturating_add(bptt) < loops;
            if early {
                state = updated.detach();
                aux = Some(add_aux(aux, pass_aux.detach()));
            } else {
                state = updated;
                aux = Some(add_aux(aux, pass_aux));
            }
        }
        scale_slots(&mut slot_sum, 1.0 / loops as f32);
        let aux = aux.unwrap_or_else(|| Tensor::<1>::zeros([1], &state.device()));
        (state, aux, slot_sum)
    }

    fn apply_core(
        &self,
        mut hidden: Tensor<3>,
        force_routes: Option<Tensor<3, Int>>,
    ) -> (Tensor<3>, Tensor<1>, Vec<Vec<f32>>) {
        let mut aux: Option<Tensor<1>> = None;
        let mut slots = Vec::with_capacity(self.core.len());
        for layer in &self.core {
            let (next, layer_aux, layer_slots) = layer.forward(hidden, force_routes.clone());
            hidden = next;
            aux = Some(add_aux(aux, layer_aux));
            slots.push(vector_f32(layer_slots.detach()));
        }
        let aux = aux.unwrap_or_else(|| Tensor::<1>::zeros([1], &hidden.device()));
        (hidden, aux, slots)
    }

    /// Number of stored scalar parameters, matching the census when embeddings are tied.
    pub fn parameter_elements(&self) -> u64 {
        struct Counter(u64);
        impl ModuleVisitor for Counter {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
                let elements: u64 = param
                    .val()
                    .dims()
                    .iter()
                    .map(|dim| u64::try_from(*dim).expect("dim fits u64"))
                    .product();
                self.0 += elements;
            }
        }
        let mut counter = Counter(0);
        self.visit(&mut counter);
        counter.0
    }

    /// Names of parameters whose gradient is missing or entirely zero.
    pub fn parameters_missing_gradient(&self, grads: &burn::tensor::Gradients) -> Vec<String> {
        struct Check<'a> {
            grads: &'a burn::tensor::Gradients,
            path: Vec<String>,
            missing: Vec<String>,
        }
        impl ModuleVisitor for Check<'_> {
            fn enter_module(&mut self, name: &str, _container: &str) {
                self.path.push(name.to_string());
            }
            fn exit_module(&mut self, _name: &str, _container: &str) {
                self.path.pop();
            }
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
                let name = if self.path.is_empty() {
                    "parameter".to_string()
                } else {
                    self.path.join(".")
                };
                let value = param.val();
                match value.grad(self.grads) {
                    None => self.missing.push(format!("{name}: no gradient")),
                    Some(grad) => {
                        let values = grad.into_data().try_into_vec::<f32>().expect("grad f32");
                        if values.iter().all(|entry| *entry == 0.0) {
                            self.missing.push(format!("{name}: zero gradient"));
                        }
                    }
                }
            }
        }
        let mut check = Check {
            grads,
            path: Vec::new(),
            missing: Vec::new(),
        };
        self.visit(&mut check);
        check.missing
    }

    /// One SGD step. The returned module is a fresh leaf and still requires gradients.
    pub fn sgd_step(self, grads: &burn::tensor::Gradients, learning_rate: f64) -> Self {
        struct Step<'a> {
            grads: &'a burn::tensor::Gradients,
            learning_rate: f64,
        }
        impl ModuleMapper for Step<'_> {
            fn map_float<const D: usize>(
                &mut self,
                param: burn::module::Param<Tensor<D>>,
            ) -> burn::module::Param<Tensor<D>> {
                let value = param.val();
                let Some(grad) = value.grad(self.grads) else {
                    return param;
                };
                let device = value.device();
                let updated = (value.detach() - grad.mul_scalar(self.learning_rate)).into_data();
                let updated = Tensor::from_data(updated, &device).require_grad();
                param.map(|_| updated)
            }
        }
        let mut step = Step {
            grads,
            learning_rate,
        };
        self.map(&mut step)
    }
}

fn dense_stack(layers: &[DenseBlock], mut hidden: Tensor<3>) -> Tensor<3> {
    for layer in layers {
        hidden = layer.forward(hidden);
    }
    hidden
}

fn tied_logits(hidden: Tensor<3>, embed: &Embedding) -> Tensor<3> {
    let [batch, seq, d_model] = hidden.dims();
    let weight = embed.weight.val().transpose();
    hidden
        .reshape([batch * seq, d_model])
        .matmul(weight)
        .reshape([batch, seq, embed.weight.dims()[0]])
}

fn add_slots(acc: &mut [Vec<f32>], next: &[Vec<f32>]) {
    for (layer, extra) in acc.iter_mut().zip(next) {
        for (value, add) in layer.iter_mut().zip(extra) {
            *value += *add;
        }
    }
}

fn scale_slots(slots: &mut [Vec<f32>], scale: f32) {
    for layer in slots {
        for value in layer {
            *value *= scale;
        }
    }
}

fn vector_f32(tensor: Tensor<1>) -> Vec<f32> {
    tensor
        .cast(burn::tensor::DType::F32)
        .into_data()
        .try_into_vec()
        .expect("expert slot fractions are f32")
}

/// Greedy continuation. `ids` is the prompt; returned ids include the prompt and the new tokens.
pub fn greedy_continue(
    model: &LoopMind,
    mut ids: Vec<u32>,
    loops: usize,
    max_new: usize,
    eos: u32,
    device: &burn::tensor::Device,
) -> Vec<u32> {
    let context = model.config.context_length.max(1);
    for _ in 0..max_new {
        if ids.is_empty() {
            break;
        }
        let window = if ids.len() > context {
            &ids[ids.len() - context..]
        } else {
            ids.as_slice()
        };
        let flat: Vec<i32> = window.iter().map(|id| *id as i32).collect();
        let tokens = Tensor::<2, Int>::from_data(TensorData::new(flat, [1, window.len()]), device);
        let out = model.forward(tokens, loops);
        let last = out
            .logits
            .narrow(1, window.len() - 1, 1)
            .reshape([1, model.config.vocab_size]);
        let next = last
            .argmax(1)
            .into_data()
            .try_into_vec::<i32>()
            .expect("argmax id")[0] as u32;
        ids.push(next);
        if next == eos {
            break;
        }
    }
    ids
}

fn add_aux(sum: Option<Tensor<1>>, next: Tensor<1>) -> Tensor<1> {
    match sum {
        Some(sum) => sum + next,
        None => next,
    }
}

/// Write tiny-config weights and Rust logits for `python/parity.py`.
pub fn export_parity_fixture(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let config = LoopMindConfig::tiny();
    let device = burn::tensor::Device::flex();
    let mut model = LoopMind::init(&config, &device);
    let tensors = assign_parity_weights(&mut model, dir)?;
    let inputs = vec![vec![1, 5, 7, 2, 9], vec![0, 3, 4, 8, 6]];
    let flat: Vec<i32> = inputs.iter().flatten().copied().collect();
    let tokens = Tensor::<2, Int>::from_data(
        TensorData::new(flat, [inputs.len(), inputs[0].len()]),
        &device,
    );
    let mut logits = Vec::new();
    let mut aux = Vec::new();
    for loops in [1usize, 3] {
        let out = model.forward(tokens.clone(), loops);
        let shape = out.logits.dims();
        let values = out
            .logits
            .into_data()
            .try_into_vec::<f32>()
            .expect("logits");
        let file = format!("logits_r{loops}.f32");
        write_f32(&dir.join(&file), &values)?;
        logits.push(TensorFile {
            name: format!("logits_r{loops}"),
            shape: shape.as_slice().to_vec(),
            file,
        });
        let aux_value = out.aux_loss.into_data().try_into_vec::<f32>().expect("aux");
        aux.push(aux_value[0]);
    }
    let manifest = Manifest {
        config,
        inputs,
        loops: vec![1, 3],
        tensors,
        logits,
        aux,
    };
    std::fs::write(
        dir.join("manifest.toml"),
        toml::to_string_pretty(&manifest).expect("manifest toml"),
    )
}

fn assign_parity_weights(model: &mut LoopMind, dir: &Path) -> std::io::Result<Vec<TensorFile>> {
    let mut tensors = Vec::new();
    let mut counter = 0u64;
    overwrite(
        &mut model.embed.weight,
        "embed.weight",
        dir,
        &mut tensors,
        &mut counter,
        Fill::Weight,
    )?;
    for (index, block) in model.prelude.iter_mut().enumerate() {
        overwrite_dense(
            block,
            &format!("prelude.{index}"),
            dir,
            &mut tensors,
            &mut counter,
        )?;
    }
    for (index, block) in model.core.iter_mut().enumerate() {
        overwrite_moe(
            block,
            &format!("core.{index}"),
            dir,
            &mut tensors,
            &mut counter,
        )?;
    }
    for (index, block) in model.coda.iter_mut().enumerate() {
        overwrite_dense(
            block,
            &format!("coda.{index}"),
            dir,
            &mut tensors,
            &mut counter,
        )?;
    }
    overwrite(
        &mut model.final_norm.gamma,
        "final_norm.gamma",
        dir,
        &mut tensors,
        &mut counter,
        Fill::Norm,
    )?;
    Ok(tensors)
}

fn overwrite_dense(
    block: &mut DenseBlock,
    prefix: &str,
    dir: &Path,
    tensors: &mut Vec<TensorFile>,
    counter: &mut u64,
) -> std::io::Result<()> {
    overwrite(
        &mut block.attn_norm.gamma,
        &format!("{prefix}.attn_norm.gamma"),
        dir,
        tensors,
        counter,
        Fill::Norm,
    )?;
    overwrite_attn(
        &mut block.attn,
        &format!("{prefix}.attn"),
        dir,
        tensors,
        counter,
    )?;
    overwrite(
        &mut block.ffn_norm.gamma,
        &format!("{prefix}.ffn_norm.gamma"),
        dir,
        tensors,
        counter,
        Fill::Norm,
    )?;
    overwrite_swiglu(
        &mut block.ffn,
        &format!("{prefix}.ffn"),
        dir,
        tensors,
        counter,
    )
}

fn overwrite_moe(
    block: &mut MoeBlock,
    prefix: &str,
    dir: &Path,
    tensors: &mut Vec<TensorFile>,
    counter: &mut u64,
) -> std::io::Result<()> {
    overwrite(
        &mut block.attn_norm.gamma,
        &format!("{prefix}.attn_norm.gamma"),
        dir,
        tensors,
        counter,
        Fill::Norm,
    )?;
    overwrite_attn(
        &mut block.attn,
        &format!("{prefix}.attn"),
        dir,
        tensors,
        counter,
    )?;
    overwrite(
        &mut block.ffn_norm.gamma,
        &format!("{prefix}.ffn_norm.gamma"),
        dir,
        tensors,
        counter,
        Fill::Norm,
    )?;
    overwrite(
        &mut block.moe.router.weight,
        &format!("{prefix}.moe.router.weight"),
        dir,
        tensors,
        counter,
        Fill::Weight,
    )?;
    for (index, expert) in block.moe.experts.iter_mut().enumerate() {
        overwrite_swiglu(
            expert,
            &format!("{prefix}.moe.experts.{index}"),
            dir,
            tensors,
            counter,
        )?;
    }
    Ok(())
}

fn overwrite_attn(
    attn: &mut crate::blocks::Attention,
    prefix: &str,
    dir: &Path,
    tensors: &mut Vec<TensorFile>,
    counter: &mut u64,
) -> std::io::Result<()> {
    overwrite(
        &mut attn.q.weight,
        &format!("{prefix}.q.weight"),
        dir,
        tensors,
        counter,
        Fill::Weight,
    )?;
    overwrite(
        &mut attn.k.weight,
        &format!("{prefix}.k.weight"),
        dir,
        tensors,
        counter,
        Fill::Weight,
    )?;
    overwrite(
        &mut attn.v.weight,
        &format!("{prefix}.v.weight"),
        dir,
        tensors,
        counter,
        Fill::Weight,
    )?;
    overwrite(
        &mut attn.o.weight,
        &format!("{prefix}.o.weight"),
        dir,
        tensors,
        counter,
        Fill::Weight,
    )
}

fn overwrite_swiglu(
    ffn: &mut crate::blocks::SwiGlu,
    prefix: &str,
    dir: &Path,
    tensors: &mut Vec<TensorFile>,
    counter: &mut u64,
) -> std::io::Result<()> {
    overwrite(
        &mut ffn.gate.weight,
        &format!("{prefix}.gate.weight"),
        dir,
        tensors,
        counter,
        Fill::Weight,
    )?;
    overwrite(
        &mut ffn.up.weight,
        &format!("{prefix}.up.weight"),
        dir,
        tensors,
        counter,
        Fill::Weight,
    )?;
    overwrite(
        &mut ffn.down.weight,
        &format!("{prefix}.down.weight"),
        dir,
        tensors,
        counter,
        Fill::Weight,
    )
}

#[derive(Clone, Copy)]
enum Fill {
    Weight,
    Norm,
}

fn overwrite<const D: usize>(
    slot: &mut burn::module::Param<Tensor<D>>,
    name: &str,
    dir: &Path,
    tensors: &mut Vec<TensorFile>,
    counter: &mut u64,
    fill: Fill,
) -> std::io::Result<()> {
    let current = slot.clone();
    let value = current.val();
    let dims = value.dims();
    let count = dims.iter().copied().product::<usize>();
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        *counter += 1;
        let wave = (*counter as f32 * 0.173).sin();
        data.push(match fill {
            Fill::Weight => wave * 0.05,
            Fill::Norm => 1.0 + wave * 0.01,
        });
    }
    write_f32(&dir.join(format!("{name}.f32")), &data)?;
    tensors.push(TensorFile {
        name: name.to_string(),
        shape: dims.as_slice().to_vec(),
        file: format!("{name}.f32"),
    });
    let tensor = Tensor::<D>::from_data(TensorData::new(data, dims), &value.device());
    *slot = current.map(|_| tensor);
    Ok(())
}

fn write_f32(path: &Path, values: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

#[derive(Serialize)]
struct Manifest {
    config: LoopMindConfig,
    inputs: Vec<Vec<i32>>,
    loops: Vec<usize>,
    tensors: Vec<TensorFile>,
    logits: Vec<TensorFile>,
    aux: Vec<f32>,
}

#[derive(Serialize)]
struct TensorFile {
    name: String,
    shape: Vec<usize>,
    file: String,
}
