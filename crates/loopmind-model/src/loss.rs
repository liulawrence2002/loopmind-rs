//! Token cross-entropy on tied-head logits.

use burn::tensor::activation::log_softmax;
use burn::tensor::{Int, Tensor};

/// Mean next-token loss. `logits` is `[batch, seq, vocab]` and `targets` is `[batch, seq]`.
pub fn token_cross_entropy(logits: Tensor<3>, targets: Tensor<2, Int>) -> Tensor<1> {
    let [batch, seq, _] = logits.dims();
    let log_probs = log_softmax(logits, 2);
    let gathered = log_probs.gather(2, targets.unsqueeze_dim::<3>(2));
    gathered.reshape([batch * seq]).mean().neg()
}

/// Next-token loss. Position `t` predicts token `t + 1`.
pub fn shifted_token_cross_entropy(logits: Tensor<3>, tokens: Tensor<2, Int>) -> Tensor<1> {
    let seq = logits.dims()[1];
    assert!(seq >= 2, "shifted loss needs at least two tokens");
    let logits = logits.narrow(1, 0, seq - 1);
    let targets = tokens.narrow(1, 1, seq - 1);
    token_cross_entropy(logits, targets)
}

/// Sum of next-token NLL over supervised targets.
///
/// `mask` is the SFT loss mask on the tokens themselves: `mask[t] = 1` means token
/// `t` is trained, so it is predicted from position `t - 1`. Padding has mask 0.
/// Callers divide by the supervised-target count they want to average over.
pub fn masked_shifted_nll_sum(
    logits: Tensor<3>,
    tokens: Tensor<2, Int>,
    mask: Tensor<2, Int>,
) -> Tensor<1> {
    masked_shifted_logprob_rows(logits, tokens, mask)
        .sum()
        .neg()
}

/// Per-row sum of next-token log-probabilities over supervised targets, `[batch]`.
///
/// DPO scores a response by this sum: the log-probability of the masked tokens.
pub fn masked_shifted_logprob_rows(
    logits: Tensor<3>,
    tokens: Tensor<2, Int>,
    mask: Tensor<2, Int>,
) -> Tensor<1> {
    let [batch, seq, _] = logits.dims();
    assert!(seq >= 2, "shifted loss needs at least two tokens");
    let dtype = logits.dtype();
    let logits = logits.narrow(1, 0, seq - 1);
    let targets = tokens.narrow(1, 1, seq - 1);
    let weights = mask.narrow(1, 1, seq - 1).float().cast(dtype);
    let log_probs = log_softmax(logits, 2);
    let picked = log_probs
        .gather(2, targets.unsqueeze_dim::<3>(2))
        .reshape([batch, seq - 1]);
    (picked * weights).sum_dim(1).reshape([batch])
}

/// Per-row count of supervised targets that the argmax does not predict, `[batch]`.
///
/// A row with 0 misses is one that greedy decoding reproduces token for token, which is
/// how LAMBADA scores a continuation.
pub fn masked_shifted_greedy_misses_rows(
    logits: Tensor<3>,
    tokens: Tensor<2, Int>,
    mask: Tensor<2, Int>,
) -> Tensor<1, Int> {
    let [batch, seq, _] = logits.dims();
    assert!(seq >= 2, "shifted loss needs at least two tokens");
    let best = logits
        .narrow(1, 0, seq - 1)
        .argmax(2)
        .reshape([batch, seq - 1]);
    let targets = tokens.narrow(1, 1, seq - 1);
    let weights = mask.narrow(1, 1, seq - 1);
    (best.not_equal(targets).int() * weights)
        .sum_dim(1)
        .reshape([batch])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn greedy_misses_count_only_supervised_targets_the_argmax_gets_wrong() {
        let device = Device::flex();
        // Every position predicts token 2: logit 1 on id 2, 0 elsewhere.
        let mut logits = vec![0f32; 2 * 3 * 4];
        for position in 0..6 {
            logits[position * 4 + 2] = 1.0;
        }
        let logits = Tensor::<3>::from_data(TensorData::new(logits, [2, 3, 4]), &device);
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![0, 2, 2, 0, 2, 3], [2, 3]), &device);
        let mask =
            Tensor::<2, Int>::from_data(TensorData::new(vec![0, 1, 1, 0, 1, 1], [2, 3]), &device);
        let misses = masked_shifted_greedy_misses_rows(logits, tokens, mask)
            .into_data()
            .convert::<i64>()
            .try_into_vec::<i64>()
            .unwrap();
        assert_eq!(misses, [0, 1]);
    }

    #[test]
    fn masked_nll_counts_only_supervised_targets() {
        let device = Device::flex();
        let vocab = 4;
        // Uniform logits: every target costs ln(4).
        let logits = Tensor::<3>::zeros([1, 4, vocab], &device);
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![0, 1, 2, 3], [1, 4]), &device);
        let mask = Tensor::<2, Int>::from_data(TensorData::new(vec![1, 0, 1, 1], [1, 4]), &device);
        let sum = masked_shifted_nll_sum(logits, tokens, mask).into_scalar::<f32>();
        // Position 0 is never a target, so two targets (tokens 2 and 3) are counted.
        assert!((sum - 2.0 * (vocab as f32).ln()).abs() < 1e-5, "{sum}");
    }

    #[test]
    fn row_logprobs_split_the_masked_sum_by_row() {
        let device = Device::flex();
        let logits = Tensor::<3>::zeros([2, 3, 4], &device);
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![0, 1, 2, 0, 3, 3], [2, 3]), &device);
        let mask =
            Tensor::<2, Int>::from_data(TensorData::new(vec![0, 1, 1, 0, 0, 1], [2, 3]), &device);
        let rows = masked_shifted_logprob_rows(logits, tokens, mask)
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
        let ln4 = 4f32.ln();
        assert!((rows[0] + 2.0 * ln4).abs() < 1e-5, "{rows:?}");
        assert!((rows[1] + ln4).abs() < 1e-5, "{rows:?}");
    }
}
