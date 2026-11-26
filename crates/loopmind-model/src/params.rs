//! Parameter census derived from [`crate::LoopMindConfig`].
//!
//! Counting rules:
//! - Bias-free linear maps. RMSNorm contributes one scale vector of `d_model`.
//! - Each transformer layer has a pre-attention norm and a pre-SwiGLU norm.
//!   One final RMSNorm sits in front of the output head.
//! - Attention is grouped-query: Q and the output projection are `d_model` by
//!   `n_q_heads * head_dim`; K and V are `d_model` by `n_kv_heads * head_dim`.
//! - SwiGLU is three matrices (gate, up, down): `3 * d_model * intermediate`.
//! - A router is `d_model` by `n_experts`.
//! - Tied embeddings are stored once. An untied head adds a second `vocab * d_model`.
//! - The core is stored once, however many times the loop runs.
//! - Active parameters are one pass over the unique layers with top-k experts
//!   per core layer. Embeddings and the output head are excluded. Inactive
//!   experts are excluded. Routers are included.

use crate::config::{ConfigError, LoopMindConfig};

/// Stored and active parameter counts for one config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamBreakdown {
    /// Token embedding matrix. Shared with the head when embeddings are tied.
    pub embeddings: u64,
    /// Untied output projection. Zero when embeddings are tied.
    pub head: u64,
    /// Q, K, V, and output projections across every unique layer.
    pub attention: u64,
    /// Pre-norm scales plus the final RMSNorm.
    pub norms: u64,
    /// Dense SwiGLU weights in the prelude and coda.
    pub dense_ffn: u64,
    /// Core routers. Counted in full in both total and active.
    pub routers: u64,
    /// Every expert in the shared core.
    pub experts: u64,
    /// Experts on the top-k path of one pass through the core.
    pub active_experts: u64,
    /// All stored parameters. The shared core is included once.
    pub total: u64,
    /// Parameters touched by one token on one pass, excluding embeddings and head.
    pub active: u64,
}

impl ParamBreakdown {
    /// Experts that exist in the checkpoint and are not on the top-k path.
    pub fn inactive_experts(self) -> u64 {
        self.experts - self.active_experts
    }
}

impl LoopMindConfig {
    /// Derive the parameter census from this config's widths and layer counts.
    pub fn param_breakdown(&self) -> Result<ParamBreakdown, ConfigError> {
        self.validate()?;

        let d = dim(self.d_model)?;
        let n_q = dim(self.n_q_heads)?;
        let n_kv = dim(self.n_kv_heads)?;
        let head_dim = d / n_q;
        let q_width = mul(n_q, head_dim)?;
        let kv_width = mul(n_kv, head_dim)?;
        let attn_one = add(&[
            mul(d, q_width)?,
            mul(d, kv_width)?,
            mul(d, kv_width)?,
            mul(q_width, d)?,
        ])?;

        let n_prelude = dim(self.n_prelude_layers)?;
        let n_core = dim(self.n_core_layers)?;
        let n_coda = dim(self.n_coda_layers)?;
        let n_layers = add(&[n_prelude, n_core, n_coda])?;
        let n_dense = add(&[n_prelude, n_coda])?;

        let attention = mul(attn_one, n_layers)?;
        let layer_norms = mul(mul(2, n_layers)?, d)?;
        let final_norm = if n_layers == 0 { 0 } else { d };
        let norms = add(&[layer_norms, final_norm])?;

        let dense_one = mul(mul(3, d)?, dim(self.dense_intermediate_size)?)?;
        let dense_ffn = mul(dense_one, n_dense)?;

        let n_experts = dim(self.n_experts)?;
        let router_one = mul(d, n_experts)?;
        let routers = mul(router_one, n_core)?;

        let expert = mul(mul(3, d)?, dim(self.expert_intermediate_size)?)?;
        let experts = mul(mul(expert, n_experts)?, n_core)?;
        let active_experts = mul(mul(expert, dim(self.experts_per_token)?)?, n_core)?;

        let embeddings = mul(dim(self.vocab_size)?, d)?;
        let head = if self.tie_embeddings { 0 } else { embeddings };

        let total = add(&[
            embeddings, head, attention, norms, dense_ffn, routers, experts,
        ])?;
        let active = add(&[attention, norms, dense_ffn, routers, active_experts])?;

        Ok(ParamBreakdown {
            embeddings,
            head,
            attention,
            norms,
            dense_ffn,
            routers,
            experts,
            active_experts,
            total,
            active,
        })
    }

    /// Multi-line census for the `params` command.
    pub fn report(&self) -> Result<String, ConfigError> {
        let params = self.param_breakdown()?;
        Ok(render(self, &params))
    }
}

fn render(config: &LoopMindConfig, params: &ParamBreakdown) -> String {
    let backends = crate::compiled_backends().join(", ");
    let mut out = String::new();
    line(&mut out, format_args!("loopmind parameter census"));
    line(
        &mut out,
        format_args!("framework: {} ({backends})", crate::framework_name()),
    );
    line(
        &mut out,
        format_args!(
            "d_model: {}   q_heads: {}   kv_heads: {}   head_dim: {}",
            config.d_model,
            config.n_q_heads,
            config.n_kv_heads,
            config.head_dim()
        ),
    );
    line(
        &mut out,
        format_args!(
            "rope_theta: {}   rms_norm_eps: {}",
            config.rope_theta, config.rms_norm_eps
        ),
    );
    line(
        &mut out,
        format_args!(
            "vocab: {} {}   context: {}",
            config.vocab_size,
            if config.tie_embeddings {
                "tied"
            } else {
                "untied"
            },
            config.context_length
        ),
    );
    line(
        &mut out,
        format_args!(
            "layers: {} dense prelude + {} moe core (looped) + {} dense coda",
            config.n_prelude_layers, config.n_core_layers, config.n_coda_layers
        ),
    );
    line(
        &mut out,
        format_args!(
            "experts: {} x width {}, top-{}, re-route each pass: {}",
            config.n_experts,
            config.expert_intermediate_size,
            config.experts_per_token,
            config.reroute_every_pass
        ),
    );
    line(
        &mut out,
        format_args!(
            "loops: mean {}, max {}, backprop through last {}",
            config.loop_count_mean, config.loop_count_max, config.bptt_passes
        ),
    );
    line(
        &mut out,
        format_args!("aux loss coef: {}", config.aux_loss_coef),
    );
    line(&mut out, format_args!(""));

    let embed_label = if config.tie_embeddings {
        "embeddings (tied)"
    } else {
        "embeddings"
    };
    let mut rows = vec![
        (embed_label, params.embeddings),
        ("attention", params.attention),
        ("rmsnorm", params.norms),
        ("dense swiglu", params.dense_ffn),
        ("routers", params.routers),
        ("experts (all)", params.experts),
    ];
    if params.head > 0 {
        rows.insert(1, ("output head (untied)", params.head));
    }
    rows.push(("total", params.total));
    rows.push(("active experts", params.active_experts));
    rows.push(("inactive experts", params.inactive_experts()));
    rows.push(("active per token", params.active));
    push_table(&mut out, &rows);

    line(&mut out, format_args!(""));
    line(
        &mut out,
        format_args!(
            "active excludes embeddings and the output head; the shared core is counted once"
        ),
    );
    line(
        &mut out,
        format_args!("active = attention + rmsnorm + dense swiglu + routers + active experts"),
    );
    line(
        &mut out,
        format_args!("total = active + inactive experts + embeddings + untied head"),
    );
    out
}

fn push_table(out: &mut String, rows: &[(&str, u64)]) {
    let label_width = rows.iter().map(|(label, _)| label.len()).max().unwrap_or(0);
    let value_width = rows
        .iter()
        .map(|(_, value)| format_count(*value).len())
        .max()
        .unwrap_or(0);
    for (label, value) in rows {
        line(
            out,
            format_args!(
                "  {label:<label_width$}  {value:>value_width$}",
                value = format_count(*value)
            ),
        );
    }
}

fn format_count(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn line(out: &mut String, args: std::fmt::Arguments<'_>) {
    std::fmt::Write::write_fmt(out, args).expect("formatting into a string");
    out.push('\n');
}

fn dim(value: usize) -> Result<u64, ConfigError> {
    u64::try_from(value).map_err(|_| ConfigError::Invalid("dimension does not fit in u64".into()))
}

fn mul(left: u64, right: u64) -> Result<u64, ConfigError> {
    left.checked_mul(right)
        .ok_or_else(|| ConfigError::Invalid("parameter count overflowed".into()))
}

fn add(values: &[u64]) -> Result<u64, ConfigError> {
    values.iter().try_fold(0u64, |acc, value| {
        acc.checked_add(*value)
            .ok_or_else(|| ConfigError::Invalid("parameter count overflowed".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_census_matches_hand_expansion() {
        let config = LoopMindConfig::default();
        let params = config.param_breakdown().unwrap();

        let d = 256u64;
        let attn_one = d * d + d * 128 + d * 128 + d * d;
        assert_eq!(attn_one, 196_608);
        let attention = attn_one * 4;
        let norms = 4 * 2 * d + d;
        let dense = 2 * 3 * d * 768;
        let routers = 2 * d * 32;
        let expert = 3 * d * 128;
        let experts = 2 * 32 * expert;
        let active_experts = 2 * 2 * expert;
        let embeddings = 6_400 * d;

        assert_eq!(params.attention, attention);
        assert_eq!(params.norms, norms);
        assert_eq!(params.dense_ffn, dense);
        assert_eq!(params.routers, routers);
        assert_eq!(params.experts, experts);
        assert_eq!(params.active_experts, active_experts);
        assert_eq!(params.embeddings, embeddings);
        assert_eq!(params.head, 0);
        assert_eq!(
            params.total,
            embeddings + attention + norms + dense + routers + experts
        );
        assert_eq!(
            params.active,
            attention + norms + dense + routers + active_experts
        );
        assert_eq!(params.total, 9_914_624);
        assert_eq!(params.active, 2_377_984);

        // The M0 brief's 9_654_016 / 2_117_376 uses the same embedding and MoE
        // terms. Both figures are short by the dense-block residual below.
        // No integer dense width hits that residual for this pre-norm block.
        assert_eq!(params.total - 9_654_016, params.active - 2_117_376);
        assert_eq!(
            params.total - params.active - embeddings,
            params.inactive_experts()
        );
    }

    #[test]
    fn loop_settings_and_context_do_not_add_parameters() {
        let base = LoopMindConfig::default().param_breakdown().unwrap();
        let shifted = LoopMindConfig {
            context_length: 128,
            loop_count_mean: 1.0,
            loop_count_max: 4,
            bptt_passes: 1,
            reroute_every_pass: false,
            rope_theta: 10_000.0,
            ..LoopMindConfig::default()
        };
        assert_eq!(shifted.param_breakdown().unwrap(), base);
    }

    #[test]
    fn untying_the_head_adds_an_embedding_sized_matrix_to_the_total_only() {
        let tied = LoopMindConfig::default().param_breakdown().unwrap();
        let untied = LoopMindConfig {
            tie_embeddings: false,
            ..LoopMindConfig::default()
        };
        let untied = untied.param_breakdown().unwrap();
        assert_eq!(untied.head, tied.embeddings);
        assert_eq!(untied.total, tied.total + tied.embeddings);
        assert_eq!(untied.active, tied.active);
    }

    #[test]
    fn routing_all_experts_grows_active_without_growing_the_checkpoint() {
        let sparse = LoopMindConfig::default().param_breakdown().unwrap();
        let dense_route = LoopMindConfig {
            experts_per_token: LoopMindConfig::default().n_experts,
            ..LoopMindConfig::default()
        };
        let dense_route = dense_route.param_breakdown().unwrap();
        assert_eq!(dense_route.total, sparse.total);
        assert_eq!(dense_route.active_experts, dense_route.experts);
        assert!(dense_route.active > sparse.active);
        assert_eq!(
            dense_route.active - sparse.active,
            sparse.inactive_experts()
        );
    }

    #[test]
    fn report_prints_the_default_totals() {
        let config = LoopMindConfig::default();
        let report = config.report().unwrap();
        assert!(report.contains("9,914,624"));
        assert!(report.contains("2,377,984"));
        assert!(report.contains("burn"));
        assert!(report.contains("flex"));
        assert!(report.contains("embeddings (tied)"));
    }

    #[test]
    fn format_count_groups_thousands() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(512), "512");
        assert_eq!(format_count(2_048), "2,048");
        assert_eq!(format_count(1_638_400), "1,638,400");
        assert_eq!(format_count(9_914_624), "9,914,624");
    }
}
