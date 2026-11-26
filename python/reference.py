"""PyTorch reference for the LoopMind forward pass.

Linear weights use Burn's layout, ``y = x @ weight`` with ``weight`` shaped
``[d_input, d_output]``. This is not ``torch.nn.Linear``.

The recurrence matches ``loopmind-model``:

    e = Prelude(Embed(tokens))
    s = e
    for _ in range(r):
        s = Core(s) + e
    logits = FinalRMSNorm(Coda(s)) @ E.transpose(-1, -2)

Run ``python python/reference.py target/parity`` after
``cargo test -p loopmind-model export_parity_fixture``.
"""

from __future__ import annotations

import argparse
import struct
import tomllib
from pathlib import Path

import torch

CAUSAL_FILL = -1.0e9
TOPK_EPS = 1.0e-9


def load_f32(path: Path, shape: list[int]) -> torch.Tensor:
    raw = path.read_bytes()
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x @ weight


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * gamma


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return torch.cat((-second, first), dim=-1)


def rope_tables(seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
    inv_freq = torch.pow(torch.tensor(theta, dtype=torch.float64), -exponents).float()
    positions = torch.arange(seq, dtype=torch.float32)
    angles = positions[:, None] * inv_freq[None, :]
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat((cos_half, cos_half), dim=-1)[None, None, :, :]
    sin = torch.cat((sin_half, sin_half), dim=-1)[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, kv_heads, seq, head = x.shape
    return (
        x.unsqueeze(2)
        .repeat(1, 1, n_rep, 1, 1)
        .reshape(batch, kv_heads * n_rep, seq, head)
    )


def attention(x: torch.Tensor, weights: dict[str, torch.Tensor], config: dict) -> torch.Tensor:
    batch, seq, _ = x.shape
    n_q = int(config["n_q_heads"])
    n_kv = int(config["n_kv_heads"])
    head = int(config["d_model"]) // n_q

    def as_heads(projected: torch.Tensor, heads: int) -> torch.Tensor:
        return projected.reshape(batch, seq, heads, head).permute(0, 2, 1, 3)

    q = as_heads(linear(x, weights["q"]), n_q)
    k = as_heads(linear(x, weights["k"]), n_kv)
    v = as_heads(linear(x, weights["v"]), n_kv)
    cos, sin = rope_tables(seq, head, float(config["rope_theta"]))
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    n_rep = n_q // n_kv
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    scores = (q @ k.transpose(-1, -2)) / (head**0.5)
    causal = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, CAUSAL_FILL)
    context = torch.softmax(scores, dim=-1) @ v
    context = context.permute(0, 2, 1, 3).reshape(batch, seq, n_q * head)
    return linear(context, weights["o"])


def swiglu(
    x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    return linear(silu(linear(x, gate)) * linear(x, up), down)


def mixture(
    x: torch.Tensor,
    router: torch.Tensor,
    experts: list[dict[str, torch.Tensor]],
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq, d_model = x.shape
    flat = x.reshape(batch * seq, d_model)
    probs = torch.softmax(linear(flat, router), dim=-1)
    topk_w, topk_i = torch.topk(probs, top_k, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp_min(TOPK_EPS)
    n_experts = router.shape[1]
    assignment = torch.nn.functional.one_hot(topk_i, n_experts).to(flat.dtype)
    gates = (assignment * topk_w.unsqueeze(-1)).sum(dim=1)
    mixed = torch.zeros_like(flat)
    for expert_id, expert in enumerate(experts):
        mixed = mixed + swiglu(flat, expert["gate"], expert["up"], expert["down"]) * gates[:, expert_id : expert_id + 1]
    tokens = float(probs.shape[0])
    slot_fraction = assignment.sum(dim=1).sum(dim=0) / (tokens * top_k)
    aux = n_experts * (slot_fraction * probs.mean(dim=0)).sum()
    return mixed.reshape(batch, seq, d_model), aux


def dense_block(x: torch.Tensor, block: dict, config: dict) -> torch.Tensor:
    eps = float(config["rms_norm_eps"])
    x = x + attention(rms_norm(x, block["attn_norm"], eps), block["attn"], config)
    normed = rms_norm(x, block["ffn_norm"], eps)
    ffn = block["ffn"]
    return x + swiglu(normed, ffn["gate"], ffn["up"], ffn["down"])


def moe_block(x: torch.Tensor, block: dict, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    eps = float(config["rms_norm_eps"])
    x = x + attention(rms_norm(x, block["attn_norm"], eps), block["attn"], config)
    normed = rms_norm(x, block["ffn_norm"], eps)
    mixed, aux = mixture(normed, block["router"], block["experts"], int(config["experts_per_token"]))
    return x + mixed, aux


def forward(token_ids: torch.Tensor, weights: dict, config: dict, loops: int) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = weights["embed"][token_ids]
    for block in weights["prelude"]:
        hidden = dense_block(hidden, block, config)
    prelude = hidden
    state = prelude
    aux = torch.zeros((), dtype=torch.float32)
    for _ in range(loops):
        core_hidden = state
        pass_aux = torch.zeros((), dtype=torch.float32)
        for block in weights["core"]:
            core_hidden, layer_aux = moe_block(core_hidden, block, config)
            pass_aux = pass_aux + layer_aux
        state = core_hidden + prelude
        aux = aux + pass_aux
    for block in weights["coda"]:
        state = dense_block(state, block, config)
    state = rms_norm(state, weights["final_norm"], float(config["rms_norm_eps"]))
    logits = state @ weights["embed"].transpose(0, 1)
    return logits, aux * float(config["aux_loss_coef"])


def _group(tensors: dict[str, torch.Tensor], prefix: str, kind: str) -> dict:
    attn = {
        proj: tensors[f"{prefix}.attn.{proj}.weight"]
        for proj in ("q", "k", "v", "o")
    }
    block = {
        "attn_norm": tensors[f"{prefix}.attn_norm.gamma"],
        "attn": attn,
        "ffn_norm": tensors[f"{prefix}.ffn_norm.gamma"],
    }
    if kind == "dense":
        block["ffn"] = {
            proj: tensors[f"{prefix}.ffn.{proj}.weight"]
            for proj in ("gate", "up", "down")
        }
    else:
        expert_ids = sorted(
            {
                int(name.split(".")[4])
                for name in tensors
                if name.startswith(f"{prefix}.moe.experts.")
            }
        )
        block["router"] = tensors[f"{prefix}.moe.router.weight"]
        block["experts"] = [
            {
                proj: tensors[f"{prefix}.moe.experts.{index}.{proj}.weight"]
                for proj in ("gate", "up", "down")
            }
            for index in expert_ids
        ]
    return block


def load_weights(directory: Path, manifest: dict) -> dict:
    tensors = {
        entry["name"]: load_f32(directory / entry["file"], entry["shape"])
        for entry in manifest["tensors"]
    }
    config = manifest["config"]
    return {
        "embed": tensors["embed.weight"],
        "prelude": [_group(tensors, f"prelude.{i}", "dense") for i in range(config["n_prelude_layers"])],
        "core": [_group(tensors, f"core.{i}", "moe") for i in range(config["n_core_layers"])],
        "coda": [_group(tensors, f"coda.{i}", "dense") for i in range(config["n_coda_layers"])],
        "final_norm": tensors["final_norm.gamma"],
    }


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left - right).abs().max().item()


def check(directory: Path, atol: float, rtol: float) -> None:
    manifest = tomllib.loads((directory / "manifest.toml").read_text())
    config = manifest["config"]
    weights = load_weights(directory, manifest)
    token_ids = torch.tensor(manifest["inputs"], dtype=torch.long)
    worst = 0.0
    for index, loops in enumerate(manifest["loops"]):
        logits, aux = forward(token_ids, weights, config, int(loops))
        entry = manifest["logits"][index]
        reference = load_f32(directory / entry["file"], entry["shape"])
        diff = max_abs(logits, reference)
        scale = reference.abs().max().item()
        limit = atol + rtol * scale
        aux_diff = abs(aux.item() - manifest["aux"][index])
        print(
            f"r={loops} max_abs={diff:.3e} atol+rtol*max={limit:.3e} aux_abs={aux_diff:.3e}"
        )
        worst = max(worst, diff, aux_diff)
        if diff > limit or aux_diff > limit:
            raise SystemExit(f"parity failed for r={loops}")
    print(f"parity ok, worst abs diff {worst:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    check(args.directory, args.atol, args.rtol)


if __name__ == "__main__":
    main()
