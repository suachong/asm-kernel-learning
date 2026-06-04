# Benchmark Walkthrough: What `co_compare.cpp` Actually Measures

> **Purpose:** Before you trust "+10%" or "1.40×" you need to know what's
> being compared, on what data, with what kind of error checking. This doc
> walks the C launcher line by line.
>
> **The file:** `~/ggemm-asm-fwd/co_compare.cpp` (321 lines). Identical in
> structure to `~/ggemm-asm-wgrad/co_compare.cpp`.

---

## 1. The 60-second summary

`co_compare` is a HIP C++ program that:

1. Loads two `.co` files (a **reference** and a **test** kernel).
2. For each registered call site (`gate_up_fwd`, `down_fwd`, etc.):
   - Allocates HBM buffers for `A`, `B`, `out_ref`, `out_test`, `scales`,
     `group_offsets`.
   - Fills `A` and `B` with random FP8 bytes (deterministic seed = 42).
   - Launches **both kernels on the same inputs**, writing to separate
     output buffers.
   - Compares the two outputs: **cosine similarity** + **max absolute
     diff** + counts of zeros and NaNs.
   - If `--benchmark` is set: warmup 50 iters, then time `niters` (default
     200) launches per kernel, report avg ms + TFLOPS + speedup.
3. Reports per-site, never reports a single composite number.

That's it. **No torch reference.** The test is purely "does the optimized
kernel produce numerically the same output as the reference kernel."

---

## 2. The data

`~/ggemm-asm-fwd/co_compare.cpp:158-167`

```cpp
const int E = 32;            // experts (matches MoE workload)
const int M_TOTAL = 131072;  // tokens (matches MoE training batch)

int64_t h_offs[E + 1];
int per_expert = M_TOTAL / E;
for (int i = 0; i <= E; i++) h_offs[i] = (int64_t)i * per_expert;
```

Every expert gets **exactly 4096 tokens** (`131072 / 32`). In reality the
MoE router would assign tokens non-uniformly, but the bench picks uniform
to make sites comparable. This is a **fine choice for kernel tuning** — the
ASM kernel is invariant to the group sizes as long as their sum is fixed —
but it's worth knowing that the bench number isn't the real-workload
number; the production routing distribution will differ slightly.

### The random fill (lines 216-225)

```cpp
std::mt19937 rng(42);
size_t max_sz = a_bytes > b_bytes ? a_bytes : b_bytes;
uint8_t* h_buf = (uint8_t*)malloc(max_sz);
for (size_t j = 0; j < a_bytes; j++) h_buf[j] = rng() & 0x3F;
HIP_CHECK(hipMemcpy(d_a, h_buf, a_bytes, hipMemcpyHostToDevice));
for (size_t j = 0; j < b_bytes; j++) h_buf[j] = rng() & 0x3F;
```

Each FP8 byte is masked with `0x3F` — so the top 2 bits are always 0.
That keeps the random fp8 values in a "safe" magnitude range (no
overflows to infinity, no edge encodings) but still varied enough to
exercise the MFMA. **Both kernels see byte-identical input** (deterministic
seed, same buffer). That's what makes the cos-similarity comparison
meaningful.

### Scales (lines 226-228)

```cpp
float scale_val = 1.0f;
HIP_CHECK(hipMemcpy(d_a_scale, &scale_val, sizeof(float), ...));
HIP_CHECK(hipMemcpy(d_b_scale, &scale_val, sizeof(float), ...));
```

`scale = 1.0` for both operands. Production code uses real per-tensor
scales; the bench short-circuits this because the kernels treat `scale` as
a single fp32 multiplier and aren't sensitive to its value beyond a
constant factor in the output.

---

## 3. The six sites

`co_compare.cpp:149-156`

```cpp
Site sites[] = {
    {"gate_up_fwd",   2880, 5760, false},
    {"down_fwd",      2880, 2880, false},
    {"down_dgrad",    2880, 2880, false},
    {"gate_up_dgrad", 5760, 2880, false},
    {"gate_up_wgrad", 2880, 5760, true},
    {"down_wgrad",    2880, 2880, true},
};
```

For FWD/DGRAD sites (`is_wgrad = false`):

- `M = 131072`, `K = D1`, `N = D2`
- `A` shape `(M, K)`, `B` shape `(E, N, K)`, `out` shape `(M, N)`
- FLOPs = `2 * M * N * K`

For WGRAD sites (`is_wgrad = true`):

- `M = 131072`, `OUT_M = D1`, `OUT_N = D2`
- `A` shape `(M, OUT_M)`, `B` shape `(M, OUT_N)`, `out` shape `(E, OUT_M, OUT_N)`
- FLOPs = `2 * M * OUT_M * OUT_N`

These shapes are pinned to the gpt-oss-20b MoE call sites you've been
working on. They match the screenshot you sent the vendor.

---

## 4. The launch (lines 64-98)

```cpp
HIP_CHECK(hipModuleLaunchKernel(func, num_sms, 1, 1, 512, 1, 1, 65536,
                                 nullptr, nullptr, config));
HIP_CHECK(hipDeviceSynchronize());
```

- **Grid:** `(num_sms, 1, 1)` = 256 workgroups on MI355X (one per CU). This
  is a **persistent kernel** — each workgroup processes many output tiles
  in a loop, instead of having one workgroup per tile.
- **Block:** `(512, 1, 1)` = 512 threads = 8 wavefronts (each wave is 64
  threads). For dot_scaled wgrad it's `(1024, 1, 1)` = 16 waves.
- **Shared:** 65,536 bytes of LDS allocated to each workgroup.

The `hipDeviceSynchronize` after every launch is what makes the timing
fair — the bench is wall-clock for a fully-completed kernel, not just the
launch latency.

---

## 5. The correctness check (lines 244-265)

```cpp
double dot = 0, norm_r = 0, norm_t = 0;
float max_diff = 0;
for (size_t j = 0; j < out_elems; j++) {
    // unpack bfloat16 -> float (h_ref[j] is a uint16 holding the upper
    // 16 bits of an fp32; we shift it left 16 to recover the float)
    float rv, tv;
    uint32_t r32 = (uint32_t)h_ref[j] << 16;
    uint32_t t32 = (uint32_t)h_test[j] << 16;
    memcpy(&rv, &r32, 4);
    memcpy(&tv, &t32, 4);
    if (std::isnan(rv) || std::isnan(tv)) { nan_count++; continue; }
    dot += (double)rv * tv;
    norm_r += (double)rv * rv;
    norm_t += (double)tv * tv;
    float diff = fabsf(rv - tv);
    if (diff > max_diff) max_diff = diff;
}
double cos_sim = dot / (sqrt(norm_r) * sqrt(norm_t) + 1e-12);
const char* status = cos_sim >= 0.999 ? "PASS" : "FAIL";
```

Two metrics:

- **`cos_sim`** — cosine similarity of (flattened) ref vs. test output.
  Range `[-1, +1]`. PASS is `>= 0.999`. The README's published numbers all
  show `cos = 1.000000`, which means the optimized kernel is bit-equivalent
  to the reference (the random fill is in a "well-conditioned" range
  where fp8 rounding agrees byte-for-byte).
- **`max_diff`** — the largest absolute difference between any
  corresponding elements. `0.000000` means literally bit-identical
  bfloat16 outputs.

**This is the key correctness property:** ASM scheduling never changes
which floating-point operations happen, only when. So the result must be
bit-identical to the Triton reference, **regardless of input values**. A
non-zero `max_diff` is a red flag — it means your optimization actually
changed the math (e.g. by reordering an MFMA across a dependency), and the
diff is wrong.

The README's wgrad legacy kernel shows `max_diff=0.031250` against the
torch reference because there it's comparing across MFMA opcodes; against
the Triton legacy `.co` it's `max_diff=0.000`.

---

## 6. The timing (lines 267-305)

```cpp
for (int i = 0; i < warmup; i++) launch_kernel(ref_func, ...);
HIP_CHECK(hipEventRecord(start));
for (int i = 0; i < niters; i++) launch_kernel(ref_func, ...);
HIP_CHECK(hipEventRecord(stop));
HIP_CHECK(hipEventSynchronize(stop));
float ref_ms;
HIP_CHECK(hipEventElapsedTime(&ref_ms, start, stop));
ref_ms /= niters;
// ... same for test_func ...

double ref_tflops = flops / (ref_ms * 1e-3) / 1e12;
double test_tflops = flops / (test_ms * 1e-3) / 1e12;
printf("  ref=%.3f ms (%.1f TFLOPS)  test=%.3f ms (%.1f TFLOPS)  speedup=%.4fx\n",
       ref_ms, ref_tflops, test_ms, test_tflops, ref_ms / test_ms);
```

- **Warmup (50 iters)**: throws away the first 50 launches to let the GPU
  reach steady state — initial launches see clock-ramp, DVFS effects, and
  HIP module load overhead. Without warmup the first measurement is
  unreliable.
- **Timed (200 iters)**: average wall-clock time per launch via HIP
  events. HIP events are GPU-side timestamps, not CPU-side `clock_gettime`
  — they measure the kernel's actual runtime, not host-side launch
  overhead.
- **TFLOPS computation**: `2 * M * N * K / (ms * 1e-3) / 1e12`. The "2"
  accounts for "one multiply + one add per MAC."

### What can mislead you here

1. **Single-run variance is real.** Even with warmup, run-to-run variance
   on a busy node is ±0.3-0.5%. A "+0.5%" speedup from one run is noise.
   The vendor's published numbers are stable across reruns; you should
   verify the same. Run 5×, take the median.
2. **TFLOPS isn't peak utilization.** MI355X FP8 peak is ~2,500 TFLOPS
   per GPU (approximate; check the spec sheet). The reference kernel is
   at ~1,200-1,700 TFLOPS depending on shape, i.e. 50-70% of peak. The
   gap to peak isn't all addressable — barrier overhead, occupancy
   limits, LDS latency are structural (see `~/ggemm-asm-fwd/README.md`
   line 70).
3. **The bench fixes batch size.** `M_TOTAL=131072` is the gpt-oss-20b
   MoE training batch. Different M scales the kernel differently — the
   per-iteration cost is largely fixed (loop tail, prologue) and only
   the main K-loop scales. If you re-tune for a different M, you need to
   re-bench at that M.

---

## 7. What "the vendor reproduced +1.16×" actually means

When the previous agent ran the benchmark and saw the README's numbers,
the data flow was:

1. Took `kernels/variable_k_gemm_ref.co` (Triton-compiled, ~1213 TFLOPS).
2. Took `kernels/variable_k_wgrad_mega.s` (hand-tuned source).
3. Used `tools/patch_co.py` to splice the new `.text` into the ref `.co`,
   producing `kernels/variable_k_wgrad_mega.co`.
4. Ran `co_compare ref.co opt.co --benchmark --warmup 50 --iters 200 --site
   gate_up_wgrad`.
5. Got `ref=3.587 ms, test=3.091 ms, speedup=1.16x, cos=1.000, max_diff=0`.

This means: **on the same FP8 inputs, the hand-tuned kernel produces
bit-identical bfloat16 outputs and finishes 16% faster**. That's a real,
defensible number.

It does **not** mean:

- The kernel is 16% faster on your *real workload* (different routing,
  different scales, different M).
- The kernel is 16% faster *end-to-end* in the training step (the GEMM is
  only part of a larger TransformerLayer forward).
- The kernel is universally 16% faster (it's tuned for these exact
  shapes; other shapes will see different numbers).

The end-to-end benefit shows up when you plug the optimized `.co` into
Primus-Turbo and run a full training step, which is what your earlier
ASM_CO integration was building toward.
