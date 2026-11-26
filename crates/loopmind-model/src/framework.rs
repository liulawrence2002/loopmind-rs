//! Burn is the tensor framework for training and inference.
//!
//! CI builds the pure-Rust `flex` backend. Local GPU builds add `wgpu` or
//! `cuda` (the RTX 4070 Laptop uses `cuda`). No Burn type is part of this
//! crate's public API.

/// Framework linked into this crate.
pub fn framework_name() -> &'static str {
    let linked = core::any::type_name::<burn::prelude::Tensor<2>>();
    assert!(
        linked.contains("Tensor"),
        "expected Burn's Tensor in the type name, got {linked}"
    );
    "burn"
}

/// Backends compiled into this build. `flex` is always on.
pub fn compiled_backends() -> &'static [&'static str] {
    match (cfg!(feature = "wgpu"), cfg!(feature = "cuda")) {
        (false, false) => &["flex"],
        (true, false) => &["flex", "wgpu"],
        (false, true) => &["flex", "cuda"],
        (true, true) => &["flex", "wgpu", "cuda"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burn_flex_is_the_default_build() {
        assert_eq!(framework_name(), "burn");
        assert!(compiled_backends().contains(&"flex"));
    }
}
