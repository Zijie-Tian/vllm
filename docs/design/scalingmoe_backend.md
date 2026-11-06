# ScalingMoE grouped MoE backend with zero GEMMs

`scalingmoe` is an explicit, experimental backend. It executes the complete
unquantized grouped MoE pipeline; **only the two grouped GEMM computations are
placeholders that write zeros**. It no longer zeroes the final routed output
in place of running the pipeline.

```text
native router: router_logits [T,E]
    -> native TopK: topk_ids + topk_weights
    -> moe_permute (token/expert grouping and row maps)
    -> scalingmoe_grouped_gemm_zero (GEMM1, fused gate/up weight)
    -> native SiLU(gate) * up
    -> scalingmoe_grouped_gemm_zero (GEMM2, down weight)
    -> moe_unpermute (inverse mapping, alpha multiplication, top-k reduction)
    -> routed output
```

The router, model/weight loading, attention, shared experts, and outer execution
remain on their existing paths. The unquantized weights are not repacked or
quantized. The finalizer stays `TopKWeightAndReduceNoOP` **because weighted
reduction has already executed in `moe_unpermute`**, as in Humming grouped.
There is no second multiplication by the router weights in either GEMM.

## Grouped GEMM interface

Both GEMM calls use the same registered, output-mutating Torch operator:

```python
torch.ops.vllm.scalingmoe_grouped_gemm_zero(
    output, hidden_states, weight,
    expert_offsets, permuted_idx, topk_ids, topk_weights, router_logits,
)
```

- `hidden_states`: expert-grouped rows `[T * topk, input_width]`.
- `weight`: GEMM1's fused gate/up weights or GEMM2's down weights.
- `output`: grouped result `[T * topk, output_width]`, currently filled with zero.
- `expert_offsets[E + 1]` (int64): grouped-row boundaries; an empty expert has equal
  adjacent offsets.
- `permuted_idx[T * topk]` (int32): grouped row to original flattened token/slot entry.
- `topk_ids[T, topk]`: original router-selected expert IDs.
- `topk_weights[T, topk]` (FP32): original router coefficients (alpha), unchanged.
- `router_logits[T, E]`: original floating-point gate output before TopK,
  unchanged and on the same device as the MoE input. Its original dtype is
  preserved: it need not be FP32 like alpha (the BF16 DeepSeek-V2-Lite path
  produces BF16 gate logits).

For a valid grouped row `r`, `s = permuted_idx[r]` gives token `s // topk`
and top-k slot `s % topk`; the corresponding alpha is `topk_weights.flatten()[s]`.
The raw pre-TopK gate vector for that row is `router_logits[s // topk]`; it is
distinct from the selected IDs and post-routing alpha. Both GEMMs receive the
same original router-logit tensor, selected routing tensors, and grouped-row
mapping. The backend does not clone, detach, normalize, recompute, cache, or
copy router logits to the CPU. The mapping is distinct from `inv_perm`, which
is passed to weighted unpermute. The placeholder does not read or mutate input
tensors; only `output` is written.

Permutation scratch uses vLLM's per-ubatch/lane workspace manager. Gate/up,
activation, down output and final output obey buffer lifetime requirements;
down output cannot alias the weighted-unpermute destination. No routing metadata
is copied to the CPU in the forward path.

Select it explicitly with:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smoe
test "${CONDA_DEFAULT_ENV:-}" = smoe || exit 1
: "${GPU:?one authorized physical GPU index or UUID}"
: "${MODEL:?local DeepSeek model directory}"

CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$CONDA_PREFIX/bin/python" -B \
  -m vllm.entrypoints.cli.main serve "$MODEL" \
  --moe-backend scalingmoe --dtype bfloat16 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --max-model-len 256 --enforce-eager --gpu-memory-utilization 0.9 \
  --compilation-config '{"custom_ops":["all","-rotary_embedding"]}'
```

The current path supports unquantized, single-GPU CUDA, Standard activation
layout, SiLU-and-mul, and output-side router weighting. Input-side router
weighting, EP mappings, missing/malformed raw router logits, and quantized
activation scales are rejected rather than silently changing their meaning.
It is never selected by `--moe-backend auto`.
A completed request is execution evidence only: zero GEMMs mean the generated
text is not a model correctness, quality, or useful performance result.

The command uses vLLM's PyTorch-native RoPE and sampler to avoid unrelated
FlashInfer JIT requirements. These options were needed for the local CUDA 12.4
compiler with the pinned cu129 vLLM wheel; they do not change the `scalingmoe`
backend. A compatible FlashInfer JIT toolchain can use the default RoPE/sampler.
DeepSeek-V2-Lite execution was checked in eager and O2 FULL_AND_PIECEWISE
CUDA Graph modes. The eager test-only dataflow probe compares both GEMM
arguments with the actual pre-TopK gate output; it is not a performance test.
