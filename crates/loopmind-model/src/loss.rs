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
