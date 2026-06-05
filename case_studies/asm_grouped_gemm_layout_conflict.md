# Case study: ASM_CO grouped-GEMM layout conflict

> **Why this lives here.** This is a worked example of how a **layout
> assumption baked into a hand-tuned ASM kernel** silently conflicts with
> a downstream call site that uses the opposite memory layout. The
> kernel and the integration wiring are each correct in isolation; the
> conflict only surfaces when you try to compose them. It's a
> representative class of bug you should expect when shipping ASM
> kernels into a framework that wasn't co-designed with them.
>
> The root-cause writeup below is reproduced verbatim from the
> investigation in `mlperf-training/small_llm_moe_pretraining/primus/dev/cuda_graph/`
> for posterity. Cross-references to "companion docs" point to files in
> that tree, not this repo.

---

# ASM_CO Grouped-GEMM Integration — Root-Cause Findings

> **Status:** BLOCKED on kernel/workload layout mismatch. The integration
> wiring is functionally correct; the kernel as shipped is incompatible
> with the gpt-oss-20b MoE call site's tensor layout. See "What needs to
> change" for the two ways forward.
>
> **Companion docs:** integration plan in
> `asm_grouped_gemm_integration.md` (this directory); patches in
> `patches/primus_turbo/`.

---

## TL;DR

`combined_v5.co` was tuned for `rhs` memory layout `(E, N, K)` (`trans_b=True`
semantics). The gpt-oss-20b Megatron MoE stores `w1`/`w2` as `(E, K, N)` and
calls `pt.ops.grouped_gemm_fp8(..., trans_b=False, ...)`. The
`GroupedGEMMFP8ASMCOBackend.can_handle()` gate correctly refuses, the
dispatcher raises `ValueError: User specified backend ASM_CO cannot handle
the given inputs`, and training crashes on the first forward step. Dropping
the gate is **not** a fix — it would silently produce numerically wrong
outputs because the kernel walks strides assuming the inner axis is K.

---

## Evidence chain

### 1. Failure surface

Run: `gh run view 26933122363 --log` (workflow
`gpt_oss_20b_primus_training.yaml`, runner `smci355-ccs-aus-n11-21`,
dev image with our ASM_CO patches applied). First failure on rank 0/8 at
`2026-06-04T05:41:31`, in the first MoE forward of the first iteration:

```
[fused_residual_rmsnorm] fused forward raised ValueError:
  User specified backend ASM_CO cannot handle the given inputs.
  Please check input constraints or choose a different backend.;
  falling back to original forward for all layers
```

Final Python traceback (deepest frame):

```
File ".../primus_turbo/pytorch/core/backend.py", line 412, in dispatch
  raise ValueError(
ValueError: User specified backend ASM_CO cannot handle the given inputs.
```

Call chain (verbose log, abbreviated):

```
primus.modules.trainer.megatron.pre_trainer.forward_step
  → GPTModel.forward
  → TransformerBlock.forward
  → TransformerLayer (fused_residual_rmsnorm wrapper)
  → MoELayer.forward
  → MoELayer.experts_compute
  → primus.backends...primus_turbo.SequentialMLP.forward      # line 1238
  → pt.ops.grouped_gemm_fp8(...)                              # workload call site
  → GroupedGemmFP8TensorFunc.apply
  → grouped_gemm_fp8_impl(...)
  → GroupedGEMMFP8KernelDispatcher.dispatch
  → BaseGroupedGEMMKernelDispatcher.dispatch                  # backend.py:412
  → raises ValueError because no backend's can_handle() returned True
```

### 2. Workload call site

```1238:1245:/workspace/Primus/primus/backends/megatron/core/extensions/primus_turbo.py
fc1_output = pt.ops.grouped_gemm_fp8(
    permuted_local_hidden_states,   # a: [M_total, hidden=2880]
    w1,                              # b: [E=32, hidden=2880, 2*intermediate=5760]
    tokens_per_expert,
    trans_b=False,                   # ← workload passes False
    config=quant_config.data(),
)
```

`w1` is reshaped two lines above as
`self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)` →
memory layout `(E, K, N) = (32, 2880, 5760)`, contiguous with
`stride = (K*N, N, 1)`.

### 3. Backend gate (works as designed)

```436:447:/home/suachong/Primus-Turbo/primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py
if not _asm_co_enabled("PRIMUS_TURBO_ASM_CO"):
    return False
supported = True
supported &= a.dim() == 2 and b.dim() == 3
supported &= (a.dtype, b.dtype, out_dtype) in GroupedGEMMFP8ASMCOBackend.SUPPORTED_DTYPES
supported &= granularity in GroupedGEMMFP8ASMCOBackend.SUPPORTED_GRANULARITIES
supported &= not trans_a and trans_b
k = a.shape[1]
n = b.shape[-2] if trans_b else b.shape[-1]
supported &= (k, n) in _ASM_CO_FWD_SITES
supported &= b.shape[0] == 32  # E
return supported
```

`trans_b=False` flips `not trans_a and trans_b` to `False`. Every other
check would pass: `_asm_co_enabled` is True (verified in run env
`PRIMUS_TURBO_ASM_CO=1`), `a.dim()==2 and b.dim()==3` holds, dtypes match
the supported set, `(k, n) = (2880, 5760)` is in `_ASM_CO_FWD_SITES`,
`b.shape[0] == 32`. So the **only** failing constraint is `trans_b`.

### 4. Why the gate is correct (kernel-side proof)

The kernel itself encodes `(E, N, K)` layout, not `(E, K, N)`:

```47:69:/workspace/deps/Primus-Turbo/csrc/pytorch/grouped_gemm/asm_co_grouped_gemm.cpp
// Flat kernarg buffer layout (96 bytes), identical for both kernels:
//   ...
//   60 i32 stride_lhs 64 i32 stride_rhs ...
struct alignas(8) KernArgs {
    const void *lhs;
    const void *rhs;
    void       *out;
    const void *lhs_scale;
    const void *rhs_scale;
    const void *group_offs;
    int32_t     G;
    int32_t     dim1;
    int32_t     dim2;
    int32_t     stride_lhs;
    int32_t     stride_rhs;
    ...
};
```

```121:135:/workspace/deps/Primus-Turbo/csrc/pytorch/grouped_gemm/asm_co_grouped_gemm.cpp
// FWD / DGRAD: a=(M,K) fp8, b=(E,N,K) fp8, out=(M,N). transB is expected true.
at::Tensor asm_co_grouped_gemm_fp8(...) {
    TORCH_CHECK(!transA, "asm_co fwd kernel requires transA=False");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 3, "asm_co fwd expects a=2D, b=3D");
    TORCH_CHECK(granularity == "TENSORWISE", "asm_co kernels are tensorwise-scaled only");

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t E = b.size(0);
    const int64_t N = transB ? b.size(1) : b.size(2);
```

The kernarg writes `stride_rhs = N * K` and the cpp comment is explicit:
"`b = (E, N, K)`. transB is expected true." The kernel pointer arithmetic
walks `b[expert*N*K + n*K + k]`. If you feed it the workload's
`(E, K, N)` buffer with `trans_b=False`:

- The byte offset for "expert e, output element n, reduction k" becomes
  `e*K*N + n*K + k`, but the actual data at that address is
  `w1[e, k_index=n, n_index=k]` — i.e. the kernel reads K and N
  **swapped**, plus its tile/mfma walk along the inner axis is now
  striding through the wrong dimension.
- Result: numerically garbage, not a HIP error. Training would diverge,
  not crash — the worst possible silent failure.

So the gate did exactly its job: refused a layout the kernel is not tuned
for, fail-closed, with a clear error message.

### 5. Shape table

| Call site (Megatron MoE) | Workload `b` layout (memory) | Workload `trans_b` | Kernel-expected layout | Match? |
|---|---|---|---|---|
| `fc1 = a @ w1` (gate_up FWD) | `(E=32, K=2880, N=5760)` | `False` | `(E, N=5760, K=2880)`, `trans_b=True` | ❌ K/N swapped |
| `dy @ w1.T` (gate_up DGRAD) | same `w1`, transposed via `trans_b=True` semantically | `True` | `(E, N=2880, K=5760)` (reduction over original N) | needs separate kernel variant `(5760, 2880)` |
| `fc2 = act @ w2` (down FWD) | `w2: (E, intermediate=2880, hidden=2880)` | `False` | `(E, 2880, 2880)`, `trans_b=True` | ❌ same problem (symmetric shape masks it) |

`_ASM_CO_FWD_SITES = {(2880, 5760), (2880, 2880), (5760, 2880)}` was
constructed under the assumption that all FWD/DGRAD sites would arrive
with `trans_b=True`. They don't.

### 6. Pre-flight environment (rule out red herrings)

All ruled out as causes — the failure is the `can_handle()` refusal above:

- `PRIMUS_TURBO_ASM_CO=1` — present in `docker exec` env (log line 20).
- `PRIMUS_TURBO_ASM_CO_DIR=/opt/primus_turbo_asm_co` — set by image
  `ENV`, no longer being overridden empty by the launcher (fixed in
  `small_llm_moe_pretraining/primus/dev/run_with_docker_dev.sh`,
  verified by absence of `--env=PRIMUS_TURBO_ASM_CO_DIR=` in launch
  command).
- `combined_v5.co` exists in the image at that path (confirmed via the
  `asm-smoke` container).
- `PRIMUS_TURBO_GROUPED_GEMM_BACKEND=ASM_CO` propagated correctly.
- Verbose logging enabled (`MLPERF_VERBOSE_LOGS=1`), which is the only
  reason we now see the chained Python tracebacks instead of an opaque
  `SIGTERM` from `torch.distributed.elastic`.

---

## What needs to change (pick one)

Both options require external work. Neither is a same-PR fix in
Primus-Turbo or this repo.

### Option A — Re-tune / repackage the asm kernel for `(E, K, N)` rhs (recommended)

Ask the asm-kernel author for a `combined_v5_kn.co` variant whose
inner-axis mfma walk is along **N** instead of K, with
`stride_rhs = K * N` semantics. Then in the cpp launcher:

- Add a parallel kernel-name selection path keyed on `trans_b`.
- Set `stride_rhs = K * N` when `trans_b == False`.
- Relax the python gate from `not trans_a and trans_b` to just
  `not trans_a` (both layouts now supported).

This is strictly a kernel-artifact swap + launcher dispatch tweak. Does
not touch Megatron, the optimizer, weight sharding, the FP8 recipe, or
any other backend.

**Risk:** the gate-detect / tile shape that combined_v5 was tuned for may
not be optimal for the K-inner walk; the author may need to re-tune.

### Option B — Change Megatron's expert weight layout to `(E, N, K)`

Modify the `view`/storage of `weight1`/`weight2` in
`primus_turbo.SequentialMLP` and the underlying expert parameter
construction so the natural memory layout is `(E, N, K)`, and update the
call to pass `trans_b=True`.

**Why we shouldn't lead with this:**

- Touches every grouped-GEMM backend (CK, hipBLASLt, Triton, ASM_CO) — all
  need to be re-verified for the new layout.
- Touches optimizer state and the DistOpt / FSDP / weight-sharding paths
  (the param tensor stride changes).
- Touches checkpoint loading (HF weights are stored in `(K, N)` order
  per expert).
- Risks regressing the Triton baseline numerics on the MLPerf submission
  config.

### Non-options (do not do these)

- **Drop the `trans_b` constraint in `can_handle()`.** Silent numerical
  corruption. The training loss will appear to converge for a few steps
  and then diverge; profiler traces would show the asm kernel running
  successfully, masking the bug. This is exactly the trap the gate was
  written to prevent.
- **Insert a runtime `b.transpose(-1, -2).contiguous()` in the cpp
  launcher.** That is itself a full read-write of `w1` (`32 * 2880 * 5760
  * 1B = ~507 MB`) on every forward and backward — comparable in cost to
  the GEMM itself, plus a per-call allocation. Defeats the whole point of
  swapping in a tuned kernel.
- **Re-roll the FP8 recipe to a different granularity.** The granularity
  gate isn't what's failing; the workload already uses TENSORWISE.

---

## Secondary issue (independent of ASM_CO, found incidentally)

The verbose log surfaces a pre-existing bug in the fused-residual-RMSNorm
fast path:

```
File ".../primus/backends/megatron/core/extensions/fused_residual_rmsnorm.py", line 414, in _do_fused_forward
    mlp_output_with_bias = layer.mlp(
TypeError: MoELayer.forward() got an unexpected keyword argument 'padding_mask'
```

`_do_fused_forward` calls `layer.mlp(..., padding_mask=...)`, but
Megatron's `MoELayer.forward` does not accept that kwarg. The wrapper
catches the `TypeError` at `fused_residual_rmsnorm.py:172` and falls back
to `_orig_layer_forward` at line 177. On the Triton baseline that
fallback succeeds and the TypeError is silently swallowed; with
`MLPERF_VERBOSE_LOGS=1` it now prints. It's the path that *then* runs
into our ASM_CO ValueError, which is why both errors appear in the same
traceback.

**Action:** worth a separate ticket against the Primus
`fused_residual_rmsnorm` author. Not blocking ASM_CO; fixing it does not
unblock ASM_CO either. Both bugs need fixing independently.

---

## What was verified to work on the run

Useful negative signals — these are *not* the bug:

- Image build (base + dev), AINIC bundle download, patch application.
- `combined_v5.co` and `variable_k_wgrad_mega.co` present in image at
  `PRIMUS_TURBO_ASM_CO_DIR`.
- Custom torch ops `primus_turbo_cpp_extension.asm_co_grouped_gemm_fp8`
  and `..._variable_k` registered (import does not raise).
- Dispatcher entries `BackendType.ASM_CO` and
  `BackendType.ASM_CO_VARIABLE_K` resolve.
- Env-var propagation from CI → docker run → docker exec → python is
  clean (verified by `--env=...` echo in log line 20).
- `MLPERF_VERBOSE_LOGS=1` correctly surfaces the chained traceback.

So the integration plumbing is done. The blocker is purely on the kernel
artifact ↔ workload layout contract.

---

## Suggested handoff for the next agent

1. Read this file and `asm_grouped_gemm_integration.md` (same directory)
   in full before touching anything.
2. **Do not** modify `GroupedGEMMFP8ASMCOBackend.can_handle()` to relax
   the `trans_b` check without a matching kernel-side fix. See "Non-options"
   above.
3. Open conversation with the asm-kernel author (the one who produced
   `combined_v5.co`) asking for a `(E, K, N)` rhs variant. Reference this
   document and the cpp comment at
   `csrc/pytorch/grouped_gemm/asm_co_grouped_gemm.cpp:121`.
4. While waiting, the existing Triton-baseline run on the same workflow
   is the correct comparison point for any future ASM_CO numbers — do not
   spin up more ASM_CO CI runs on the current image; they will all fail
   identically with the same `ValueError`.
5. If the kernel author proposes a layout-flexible kernel: the launcher
   change is local to `asm_co_grouped_gemm.cpp` (kernel-name selection +
   stride computation) and the gate change is one line in
   `grouped_gemm_fp8_impl.py`. Both ship as updates to the existing
   `asm_co_grouped_gemm.patch`.

---

## Provenance

| Claim | Source |
|---|---|
| First failure timestamp + traceback | `gh run view 26933122363 --log`, lines 23, 24 (rank-0 ValueError + full traceback) |
| Workload call site arguments | `primus/backends/megatron/core/extensions/primus_turbo.py:1238-1245` (inside `asm-smoke` container, image `dev-ci-test-26933122363_1082`) |
| `w1` reshape | same file, line 1230 |
| Backend gate logic | `Primus-Turbo/primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py:436-447` |
| Kernarg / stride / cpp layout comment | `csrc/pytorch/grouped_gemm/asm_co_grouped_gemm.cpp:47-69, 121-135` |
| `_ASM_CO_FWD_SITES` | `grouped_gemm_fp8_impl.py:399-403` |
| Env vars on the failing run | `gh run view 26933122363 --log`, line 20 (`docker exec --env=...`) |
| Fused-residual-RMSNorm TypeError | `gh run view 26933122363 --log`, line 24, frame at `fused_residual_rmsnorm.py:414` |
