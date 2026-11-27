use std::path::Path;

use burn::module::Module;
use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::{export_parity_fixture, token_cross_entropy, LoopMind, LoopMindConfig, TrainRng};

fn scalar(tensor: &Tensor<1>) -> f32 {
    tensor.clone().into_data().try_into_vec::<f32>().unwrap()[0]
}

fn max_abs_diff(left: &Tensor<3>, right: &Tensor<3>) -> f32 {
    let left = left.clone().into_data().try_into_vec::<f32>().unwrap();
    let right = right.clone().into_data().try_into_vec::<f32>().unwrap();
    left.into_iter()
        .zip(right)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f32::max)
}

fn tokens(batch: usize, seq: usize, device: &Device) -> Tensor<2, Int> {
    let values: Vec<i32> = (0..batch * seq)
        .map(|index| (index % 7) as i32 + 1)
        .collect();
    Tensor::<2, Int>::from_data(TensorData::new(values, [batch, seq]), device)
}

#[test]
fn forward_shapes_and_loop_count_changes_logits() {
    let device = Device::flex();
    let config = LoopMindConfig::tiny();
    let model = LoopMind::init(&config, &device);
    let inputs = tokens(2, 5, &device);
    let once = model.forward(inputs.clone(), 1);
    let thrice = model.forward(inputs, 3);
    assert_eq!(once.logits.dims(), [2, 5, config.vocab_size]);
    assert_eq!(thrice.logits.dims(), [2, 5, config.vocab_size]);
    assert_eq!(once.aux_loss.dims(), [1]);
    assert!(scalar(&once.aux_loss).is_finite());
    assert!(scalar(&thrice.aux_loss).is_finite());
    assert!(max_abs_diff(&once.logits, &thrice.logits) > 1e-5);
}

#[test]
fn truncation_keeps_the_forward_values() {
    let device = Device::flex();
    let model = LoopMind::init(&LoopMindConfig::tiny(), &device);
    let inputs = tokens(1, 4, &device);
    let mut short = model.clone();
    short.config.bptt_passes = 1;
    let mut full = model;
    full.config.bptt_passes = 6;
    let short = short.forward_train_at(inputs.clone(), 6, None);
    let full = full.forward_train_at(inputs, 6, None);
    assert!(max_abs_diff(&short.logits, &full.logits) < 1e-6);
}

#[test]
fn every_routed_parameter_receives_a_gradient() {
    let device = Device::flex().autodiff();
    device.seed(3);
    let config = LoopMindConfig {
        d_model: 32,
        n_q_heads: 4,
        n_kv_heads: 2,
        vocab_size: 16,
        context_length: 8,
        dense_intermediate_size: 32,
        n_experts: 4,
        expert_intermediate_size: 16,
        ..LoopMindConfig::tiny()
    };
    let model = LoopMind::init(&config, &device).train();
    let inputs = tokens(1, 4, &device);
    // top-2 pairs that together cover experts 0, 1, 2, and 3.
    let routes = Tensor::<3, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3, 0, 2, 1, 3], [1, 4, 2]),
        &device,
    );
    let out = model.forward_train_at(inputs.clone(), 2, Some(routes));
    let loss = token_cross_entropy(out.logits, inputs) + out.aux_loss;
    let grads = loss.backward();
    let missing = model.parameters_missing_gradient(&grads);
    assert!(
        missing.is_empty(),
        "parameters without a gradient:\n{}",
        missing.join("\n")
    );
}

#[test]
fn tiny_batch_loss_decreases() {
    let device = Device::flex().autodiff();
    device.seed(11);
    let config = LoopMindConfig {
        d_model: 32,
        n_q_heads: 4,
        n_kv_heads: 2,
        vocab_size: 16,
        context_length: 8,
        dense_intermediate_size: 32,
        n_experts: 4,
        expert_intermediate_size: 16,
        aux_loss_coef: 0.0,
        ..LoopMindConfig::tiny()
    };
    let mut model = LoopMind::init(&config, &device).train();
    let inputs = tokens(2, 4, &device);
    let mut first = 0.0;
    let mut last = 0.0;
    for step in 0..40 {
        let out = model.forward_train_at(inputs.clone(), 2, None);
        let loss = token_cross_entropy(out.logits, inputs.clone()) + out.aux_loss;
        let value = scalar(&loss);
        assert!(value.is_finite(), "loss became non-finite at step {step}");
        if step == 0 {
            first = value;
        }
        last = value;
        let grads = loss.backward();
        model = model.sgd_step(&grads, 0.15);
    }
    assert!(
        last < first * 0.85,
        "expected the tiny-batch loss to fall, started at {first} and ended at {last}"
    );
}

#[test]
fn parameter_elements_match_the_census() {
    let device = Device::flex();
    for config in [LoopMindConfig::default(), LoopMindConfig::tiny()] {
        let model = LoopMind::init(&config, &device);
        let census = config.param_breakdown().unwrap();
        assert_eq!(model.parameter_elements(), census.total);
        assert_eq!(census.head, 0);
    }
}

#[test]
fn sampled_loop_counts_stay_in_range_near_the_mean() {
    let mut rng = TrainRng::new(1);
    let samples: Vec<usize> = (0..4_000).map(|_| rng.sample_loops(4.0, 16)).collect();
    assert!(samples.iter().all(|count| (1..=16).contains(count)));
    let mean = samples.iter().sum::<usize>() as f64 / samples.len() as f64;
    assert!(
        (3.6..4.5).contains(&mean),
        "sample mean {mean} is far from 4"
    );
}

#[test]
fn export_parity_fixture_writes_logits() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/parity");
    export_parity_fixture(&dir).unwrap();
    assert!(dir.join("manifest.toml").is_file());
    assert!(dir.join("logits_r1.f32").is_file());
    assert!(dir.join("logits_r3.f32").is_file());
    assert!(dir.join("embed.weight.f32").is_file());
}

#[test]
fn fused_experts_match_the_per_expert_loop() {
    let device = Device::flex().autodiff();
    device.seed(5);
    let config = LoopMindConfig::tiny();
    let moe = crate::Mixture::new(&config, &device).train();
    let values: Vec<f32> = (0..2 * 6 * config.d_model)
        .map(|index| ((index * 37 % 101) as f32 / 50.0) - 1.0)
        .collect();
    let x = Tensor::<3>::from_data(TensorData::new(values, [2, 6, config.d_model]), &device);
    let (fused, fused_aux, fused_slots) = moe.forward(x.clone(), None);
    let (looped, looped_aux, looped_slots) = moe.forward_unfused(x, None);
    assert!(max_abs_diff(&fused, &looped) < 1e-5);
    assert!((scalar(&fused_aux) - scalar(&looped_aux)).abs() < 1e-6);
    let fused_slots = fused_slots.into_data().try_into_vec::<f32>().unwrap();
    assert_eq!(
        fused_slots,
        looped_slots.into_data().try_into_vec::<f32>().unwrap()
    );
    let fused_grads = fused.sum().backward();
    let looped_grads = looped.sum().backward();
    for expert in &moe.experts {
        for weight in [&expert.gate.weight, &expert.up.weight, &expert.down.weight] {
            let left = weight.val().grad(&fused_grads).expect("fused grad");
            let right = weight.val().grad(&looped_grads).expect("looped grad");
            let left = left.into_data().try_into_vec::<f32>().unwrap();
            let right = right.into_data().try_into_vec::<f32>().unwrap();
            let gap = left
                .iter()
                .zip(&right)
                .map(|(lhs, rhs)| (lhs - rhs).abs())
                .fold(0.0, f32::max);
            assert!(gap < 1e-5, "expert weight gradient differs by {gap}");
        }
    }
}
