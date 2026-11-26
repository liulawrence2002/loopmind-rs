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
