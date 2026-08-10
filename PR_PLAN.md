# Gemma 4 Support for Penzai — Full Implementation Plan

## Context

Gemma 4 introduces four model variants with significant architectural novelty
compared to Gemma 3. This plan adds support for all four, phased so each phase
is independently shippable. Checkpoints are loaded from the Flax format
(google-deepmind/gemma library).

### Variants at a glance

| Variant | Type | Layers | Embed | New features over Gemma 3 |
|---------|------|--------|-------|---------------------------|
| 31B | Dense | 60 | 5376 | Per-layer proj dim, partial RoPE, K=V sharing, value norm, skip scale |
| E4B | Dense+PLE | 42 | 2560 | + Per-Layer Embeddings, KV cache sharing |
| E2B | Dense+PLE | 35 | 1536 | + Per-Layer Embeddings, KV cache sharing |
| 26B-A4B | MoE | 30 | 2816 | + 128-expert MoE with top-8 routing, dual-branch FFW |

### Key architectural differences from Gemma 3

1. Different `projection_dim` for local (256) vs global (512) attention layers
2. Different `num_kv_heads` for local vs global layers (31B: 16 local / 4 global; 26B: 8/2)
3. Partial RoPE: only 25% of head dims get RoPE in global layers (`ApplyRoPEToSubset`)
4. K=V sharing: global layers share K and V projections (31B, 26B)
5. Value normalization: RMSNorm on values (in addition to Q/K norm from Gemma 3)
6. Learnable skip scale on the attention residual connection
7. Per-Layer Embeddings (E2B, E4B): separate per-layer token embedding table with gating
8. KV cache sharing (E2B, E4B): later layers reuse KV projections from earlier layers
9. MoE (26B-A4B): 128 experts with top-8 routing + dense shared MLP branch

---

## Phase A: Core Infrastructure + gemma4\_31b  ✅ DONE

All changes landed and tested (16/16 tests pass, 0 new type errors).

### A1. New config fields on `LlamalikeTransformerConfig`

**File:** `penzai/models/transformer/variants/llamalike_common.py`

Added six backward-compatible optional fields:

```python
global_projection_dim: int | None = None
global_num_kv_heads: int | None = None
global_rope_proportion: float | None = None
use_value_norm: bool = False
use_skip_scale: bool = False
k_eq_v_global: bool = False
```

### A2. Refactored `_head_info` to be parameterised

Changed `_head_info(config)` → `_head_info(num_kv_heads, query_head_multiplier)`
so it can be called with per-layer effective values. Updated both call sites.

### A3. Per-layer dimensions in `build_llamalike_attention`

After resolving `attention_type` per layer, the function now also resolves:
- `projection_dim` → `global_projection_dim` for global layers
- `effective_num_kv_heads` → `global_num_kv_heads` for global layers
- `effective_query_head_multiplier` derived from the fixed total query head count

All `Linear.from_config` calls, NamedEinsum shapes, and RoPE layers within the
function use these effective values.

### A4. Partial RoPE for global layers

When `global_rope_proportion` is set and < 1.0, global layers use
`pz.nn.ApplyRoPEToSubset` (already in `penzai/nn/embeddings.py`) instead of
`pz.nn.ApplyRoPE`. Local layers are unaffected.

### A5. Value normalization

When `use_value_norm=True`, an `RMSLayerNorm` is appended to `input_to_value`,
following the same pattern as QK norm.

### A6. K=V sharing for global layers

When `k_eq_v_global=True` and the layer is global, one `pz.nn.Linear` object is
created and referenced in both `input_to_key` and `input_to_value` pipelines —
exactly the embedding-tying pattern used for tied embeddings.

### A7. `ScaledResidual` combinator

**File:** `penzai/nn/combinators.py`

New `@struct.pytree_dataclass` layer:

```python
class ScaledResidual(layer_base.Layer):
    delta: layer_base.Layer
    scale: parameters.ParameterLike

    def __call__(self, value, **side_inputs):
        return self.delta(value, **side_inputs) + value * self.scale.value
```

Exported from `penzai/pz/nn.py`.

### A8. `ScaledResidual` in `build_llamalike_block`

When `use_skip_scale=True`, the attention residual uses `ScaledResidual` with a
learnable scalar parameter instead of `pz.nn.Residual`.

### A9. Per-layer cache axes in `sampling_mode.py`

`KVCachingTransformerLM.from_uncached` now infers `cached_axes` per attention
layer by inspecting the key `Linear`'s `output_axes`, instead of using a single
global value from metadata. Backward-compatible for uniform models.

### A10–A12. gemma4\_31b preset, checkpoint loader, docstring

- Added `gemma4_31b` to `_GEMMA_PRESETS` and `_NEEDS_GATING_TRANSPOSE`.
- Updated `preset_name` Literal type.
- Auto-detection now checks for `skip_scale` to distinguish Gemma 4 from 3.
- Checkpoint loader handles value norm, skip scale, K=V sharing, and per-layer
  attention weight shapes (branches on `attention_type[i]`).
- Updated module docstring to reference Gemma 4.

### Tests

Two new parameterised test cases (`like_gemma4_31b`) added to
`tests/models/transformer_llamalike_test.py`:
- `test_build_and_run_gemma_like_gemma4_31b` — build + forward pass via `jax.eval_shape`
- `test_build_and_run_sampling_mode_like_gemma4_31b` — sampling mode + decoding loop

All 16 tests pass (14 existing + 2 new).

---

## Phase B: gemma4\_E2B + gemma4\_E4B (Per-Layer Embeddings + KV Cache Sharing)  ✅ DONE

All changes landed and tested (18/18 tests pass, 0 new type errors).

### B1. Per-Layer Embedding layer

**File:** `penzai/models/transformer/variants/gemma.py`

New `@struct.pytree_dataclass` layer `PerLayerEmbeddingInject`:

```python
@struct.pytree_dataclass
class PerLayerEmbeddingInject(layer_base.Layer):
    """Injects per-layer embedding information into the residual stream.

    Reads token IDs from side inputs, looks up per-layer embeddings for
    the current layer, gates them with the current hidden state, projects
    back to embedding dimension, and adds to the input.

    Attributes:
        per_layer_table: Shared embedding table for all layers.
        layer_index: Which layer slice to use from the table.
        gate: Linear projection from embedding_dim to per_layer_dim (with GELU).
        projection: Linear projection from per_layer_dim back to embedding_dim.
        post_norm: RMSLayerNorm applied after projection.
        per_layer_input_dim: Dimension of per-layer embeddings.
        token_ids_input_name: Side input key for token IDs.
    """
```

**Flow per block:**
1. Read `token_ids` from side inputs
2. Index into shared PLE table: `table[{vocabulary: token_ids, layers: layer_index}]`
3. Scale by `sqrt(per_layer_input_dim)`
4. Gate: `gelu(gate_linear(hidden_state)) * ple_embedding`
5. Project back: `projection_linear(gated)`
6. Normalise and add to residual

### B2. Pass `token_ids` as side input

**File:** `penzai/models/transformer/model_parts.py`

Add `token_ids=tokens` to the side inputs in `TransformerLM.__call__`:

```python
return self.body(tokens, token_positions=token_positions,
                 token_ids=tokens, **side_inputs)
```

Backward-compatible: existing layers ignore unknown side inputs.

### B3. KV cache sharing

For E2B/E4B, the last N layers reuse K/V projections from earlier layers.

**Training mode:** Share the same `pz.nn.Linear` Python objects between source
and target layers (same pattern as embedding tying). Both layers compute the
projection; the weights are shared by identity.

**Sampling mode:** Shared-KV layers must point to the same `StateVariable` cache.
Update `sampling_mode.py` to detect when two `Attention` layers share the same
key `Linear` (by object identity or label) and wire them to the same cache.

### B4. E2B and E4B presets + checkpoint loading

```python
"gemma4_E2B": dict(
    num_decoder_blocks=35, vocab_size=262_144,
    num_kv_heads=1, query_head_multiplier=8,
    embedding_dim=1536, projection_dim=256, global_projection_dim=512,
    mlp_hidden_dim=6_144,
    per_layer_input_dim=256, num_kv_shared_layers=20,
    use_qk_norm=True, use_value_norm=True,
    use_post_attn_norm=True, use_post_ffw_norm=True,
    use_skip_scale=True, global_rope_proportion=0.25,
    rope_wavelength=1_000_000, local_rope_wavelength=10_000,
    attention_type=...,  # 5×SW(512) + 1×Global
),
"gemma4_E4B": dict(
    num_decoder_blocks=42, vocab_size=262_144,
    num_kv_heads=2, query_head_multiplier=4,
    embedding_dim=2560, projection_dim=256, global_projection_dim=512,
    mlp_hidden_dim=10_240,
    per_layer_input_dim=256, num_kv_shared_layers=18,
    use_qk_norm=True, use_value_norm=True,
    use_post_attn_norm=True, use_post_ffw_norm=True,
    use_skip_scale=True, global_rope_proportion=0.25,
    rope_wavelength=1_000_000, local_rope_wavelength=10_000,
    attention_type=...,  # 5×SW(512) + 1×Global
),
```

Checkpoint loader extended for PLE weights:
- `embedder/per_layer_embeddings/w` — `[vocab, layers, ple_dim]`
- `embedder/per_layer_model_projection/w`
- `embedder/per_layer_projection_norm/scale`
- Per block: `layer_N/per_layer_input_gate/w`, `layer_N/per_layer_projection/w`,
  `layer_N/post_per_layer_input_norm/scale`

### B5. Tests

- Build + forward pass for E2B and E4B configs via `jax.eval_shape`
- Sampling mode test with KV cache sharing
- Verify PLE layers receive token IDs and produce correct shapes

---

## Phase C: gemma4\_26b\_a4b (Mixture of Experts)  ✅ DONE

All changes landed and tested (20/20 tests pass, 0 new type errors).

### C1. MoE dataflow combinator

**File:** `penzai/nn/mixture_of_experts.py` (new)

Following the `Attention` pattern — a single `@struct.pytree_dataclass` that
orchestrates dataflow between named sub-components:

```python
@struct.pytree_dataclass
class MixtureOfExperts(layer_base.Layer):
    """A mixture-of-experts dataflow combinator.

    Routes input tokens to a subset of experts via a learned router, runs
    the selected experts, and combines their outputs with routing weights.
    Follows the same design philosophy as Attention.

    Attributes:
        input_to_routing: Maps input to (routing_weights, expert_indices).
        input_and_routing_to_output: Maps (input, routing_weights,
            expert_indices) to combined expert output.
    """
    input_to_routing: layer_base.Layer
    input_and_routing_to_output: layer_base.Layer

    def __call__(self, x, **side_inputs):
        routing_weights, expert_indices = self.input_to_routing(x, **side_inputs)
        output = self.input_and_routing_to_output(
            (x, routing_weights, expert_indices), **side_inputs
        )
        return output
```

### C2. Router and expert computation layers

**File:** `penzai/nn/mixture_of_experts.py`

**`MoETopKRouter`** — RMS standardise → learned scale → project to expert
logits → softmax → top-k → renormalise. Returns `(routing_weights, expert_indices)`.

```python
@struct.pytree_dataclass
class MoETopKRouter(layer_base.Layer):
    router_norm: layer_base.Layer
    router_scale: parameters.ParameterLike[named_axes.NamedArray]
    router_logits: layer_base.Layer
    num_selected_experts: int  # metadata, not pytree node
```

**`MoEGatedExpertComputation`** — Gathers selected expert weights, computes
GeGLU MLP per expert, scales by per-expert factors, combines with routing
weights. Expert weights stored as batched NamedArrays with `"experts"` axis.

```python
@struct.pytree_dataclass
class MoEGatedExpertComputation(layer_base.Layer):
    gating_weights: parameters.ParameterLike[named_axes.NamedArray]
    value_weights: parameters.ParameterLike[named_axes.NamedArray]
    out_weights: parameters.ParameterLike[named_axes.NamedArray]
    per_expert_scale: parameters.ParameterLike[named_axes.NamedArray]
    activation_fn: Any  # metadata, not pytree node
```

Initial implementation uses gather-based dispatch (portable, inspectable).
Can optimise with `jax.lax.ragged_dot` later.

### C3. `TransformerMoEFeedForward`

**File:** `penzai/models/transformer/model_parts.py`

```python
@pz.pytree_dataclass(has_implicitly_inherited_fields=True)
class TransformerMoEFeedForward(pz.nn.Sequential):
    """Informatively-named Sequential subclass for MoE + dense feedforward."""
```

### C4. MoE builder + block integration

**File:** `penzai/models/transformer/variants/llamalike_common.py`

New config fields:

```python
num_experts: int | None = None
num_selected_experts: int | None = None
expert_hidden_dim: int | None = None
dense_moe_shared_hidden_dim: int | None = None
```

New `build_moe_feedforward()` assembles the dual-branch structure:

```python
TransformerMoEFeedForward(sublayers=[
    pz.nn.BranchAndAddTogether(branches=[
        pz.nn.NamedGroup("dense_branch", [
            RMSLayerNorm(pre_ffw2_norm),
            build_llamalike_feedforward(mlp2, hidden=2112),
            RMSLayerNorm(post_ffw2_norm),
        ]),
        pz.nn.NamedGroup("moe_branch", [
            RMSLayerNorm(pre_ffw_norm),
            MixtureOfExperts(router, experts),
            RMSLayerNorm(post_ffw1_norm),
        ]),
    ]),
    RMSLayerNorm(post_ffw_norm),
])
```

In `build_llamalike_block`, when `config.num_experts is not None`, use
`build_moe_feedforward()` in place of `build_llamalike_feedforward()`.

### C5. 26B-A4B preset + checkpoint loading

```python
"gemma4_26b_a4b": dict(
    num_decoder_blocks=30, vocab_size=262_144,
    num_kv_heads=8, global_num_kv_heads=2, query_head_multiplier=2,
    embedding_dim=2816, projection_dim=256, global_projection_dim=512,
    mlp_hidden_dim=2_112,
    num_experts=128, num_selected_experts=8, expert_hidden_dim=704,
    dense_moe_shared_hidden_dim=2_112,
    k_eq_v_global=True,
    use_qk_norm=True, use_value_norm=True,
    use_post_attn_norm=True, use_post_ffw_norm=True,
    use_skip_scale=True, global_rope_proportion=0.25,
    rope_wavelength=1_000_000, local_rope_wavelength=10_000,
    attention_type=...,  # 5×SW(1024) + 1×Global
),
```

Checkpoint loader maps MoE weights per layer:
- `layer_N/mlp/router_logits/w` — `[embed_dim, num_experts]`
- `layer_N/mlp/router_scale` — `[embed_dim]`
- `layer_N/mlp/router_norm/scale` — `[embed_dim]`
- `layer_N/mlp/gating_einsum/w` — `[num_experts, 2, expert_dim, embed_dim]`
  (split into gating + value along dim 1)
- `layer_N/mlp/linear/w` — `[num_experts, expert_dim, embed_dim]`
- `layer_N/mlp/per_expert_scale` — `[num_experts]`
- `layer_N/mlp2/gating_einsum/w`, `layer_N/mlp2/linear/w` (dense shared MLP)

### C6. Export new layers

**File:** `penzai/pz/nn.py`

```python
from penzai.nn.mixture_of_experts import (
    MixtureOfExperts,
    MoETopKRouter,
    MoEGatedExpertComputation,
)
```

### C7. Tests

- Build + forward pass for 26B-A4B config via `jax.eval_shape`
- Sampling mode test
- Unit tests for `MoETopKRouter` and `MoEGatedExpertComputation` in isolation

---

## Files Modified (complete)

| File | Phase | Changes |
|------|-------|---------|
| `penzai/models/transformer/variants/llamalike_common.py` | A ✅, C | Config, `_head_info`, attention builder, block builder, MoE builder |
| `penzai/models/transformer/variants/gemma.py` | A ✅, B | Presets, auto-detect, checkpoint loader, PLE layers |
| `penzai/nn/combinators.py` | A ✅ | `ScaledResidual` |
| `penzai/nn/mixture_of_experts.py` | C | **New**: `MixtureOfExperts`, `MoETopKRouter`, `MoEGatedExpertComputation` |
| `penzai/models/transformer/model_parts.py` | B, C | `token_ids` side input, `TransformerMoEFeedForward` |
| `penzai/models/transformer/sampling_mode.py` | A ✅ | Per-layer cache axes |
| `penzai/pz/nn.py` | A ✅, C | Exports |
| `tests/models/transformer_llamalike_test.py` | A ✅, B, C | New test cases |

## Verification (per phase)

1. **Shape test** — Build each variant with a small test config via `jax.eval_shape`;
   verify output shape `{"batch": B, "seq": S, "vocabulary": 262144}`
2. **Forward pass** — Run with random weights and small dims; assert no NaN/shape errors
3. **Sampling mode** — Convert to `KVCachingTransformerLM`; verify per-layer cache dims
4. **Backward compat** — Existing Gemma 1/2/3 parameterised tests still pass unchanged
5. **Type check** — `pyright` on changed files; no new errors beyond pre-existing patterns
6. **Checkpoint** (if available) — Load real Flax weights, run forward pass, compare
   with reference implementation
