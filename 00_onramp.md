# ASM-Kernel-Optimization Onramp for Agentic AI Tuning

> **Goal of this doc:** Give you enough vocabulary, mental model, and pointers
> to read a diff that an AI agent proposes against a GPU assembly (`.s`) file,
> decide whether to accept it, and understand the benchmark number that comes
> back. **Not** to make you write AMDGCN assembly from scratch — that's not
> what the vendor (Mohamed Awad) was doing either.
>
> **Audience:** You usually work several abstraction layers above this — at
> the PyTorch / Megatron level. This doc closes the gap.
>
> **Companion docs in this folder:**
> - `01_terminology_primer.md` — what every term in the vendor's READMEs
>   actually means (`s_setprio`, `buffer_load hoisting`, `vmcnt`, MFMA, etc.).
> - `02_benchmark_walkthrough.md` — what `co_compare.cpp` actually measures,
>   line by line.
> - `03_roundtrip_workplan.md` — today's hands-on exercise. Start here once
>   you've skimmed the other three.
>
> Keep all four open in tabs and ping-pong between them.

---

## 1. ISA manuals — what you need, what to skim, what to skip

You do **not** need to read either manual cover to cover. They're ~800 pages
each. You need them as **references** — to look up "what does `s_setprio`
do?" when an agent proposes inserting one, and "is `v_mfma_f32_16x16x128_f8f6f4`
the right MFMA for fp8 on MI355X?" when reading the disassembly.

### The two manuals that matter for your MI355X work

| Architecture | GPU | LLVM target | When to use it |
|---|---|---|---|
| **CDNA4** | MI355X, MI350X | `gfx950` | **Primary**. Every kernel in `~/ggemm-asm-fwd` and `~/ggemm-asm-wgrad` targets this. |
| **CDNA3** | MI300X, MI325X, MI300A | `gfx942` | Cross-reference when CDNA4 manual is ambiguous (the CDNA3 manual is older and has more worked examples; many instructions are identical). |

Download links (open in your browser; this dev box's curl currently
fails against AMD's CDN with `HTTP/2 INTERNAL_ERROR` — the docs
themselves are public, just not curl-able from here):

- **CDNA4 ISA Reference Guide (MI350 series, gfx950)** —
  `https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf`
- **CDNA3 ISA Reference Guide (MI300 series, gfx942)** —
  `https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf`

Save them to `~/asm-kernel-learning/refs/cdna4-isa.pdf` and
`~/asm-kernel-learning/refs/cdna3-isa.pdf` for offline reference.

Landing pages (in case AMD reshuffles the PDF URLs):

- AMD ROCm GPU arch index: `https://rocm.docs.amd.com/en/latest/reference/gpu-arch/index.html`
- AMD GPUOpen ISA landing page: `https://gpuopen.com/amd-gpu-architecture-programming-documentation/`

### What sections to skim first (1-2 hours, not a week)

For the CDNA4 manual:

1. **Chapter 1-2** — Programming model. Wavefronts (groups of 64 threads),
   workgroups, CUs (compute units, "SMs" in NVIDIA lingo), XCDs (MI355X has
   8 XCDs each with 32 CUs = 256 CUs). Read this carefully — most
   optimization is about **hiding latency by keeping the CU busy**, and you
   can't do that without knowing what the CU is.
2. **Chapter on Memory (LDS, VMEM, scalar)** — Specifically: what `vmcnt`,
   `lgkmcnt`, `expcnt` track, and what `s_waitcnt` does. You'll see these
   on roughly every 5th line of any `.s` file.
3. **The MFMA section** — Look up `v_mfma_f32_16x16x128_f8f6f4` (the FP8
   matrix instruction used in `combined_v5.s`) and
   `v_mfma_f32_32x32x64_f8f6f4` (the dot_scaled wgrad instruction). You only
   need to understand: how many FLOPs per instruction, what's its latency
   (in cycles), and what register tuples it consumes/produces.
4. **The `buffer_load_*` family** — These are HBM (global memory) loads.
   Read the entry for `buffer_load_dwordx4`, which is what gets "hoisted"
   in both vendor optimizations.

Sections you can **skip entirely** for now:

- All branch instruction variants (the agent won't be touching these).
- Image / sampler instructions (these are graphics-only).
- Workgroup-processor mode vs. CU-mode (you're CU-mode only).
- Cooperative matrix beyond MFMA (not used in any of the four kernels here).

### Should you have the manuals open before letting an agent loose?

**Yes — but as a reference, not as prerequisite reading.** Workflow:

1. Agent proposes a diff: "I moved `buffer_load_dwordx4 v[216:223]` from
   loop-bottom to loop-top." (This is the actual wgrad optimization.)
2. You don't need to recognize that instruction by sight, but you should
   `Cmd+F` "buffer_load_dwordx4" in the CDNA4 PDF, confirm it's a 128-bit
   HBM load, and confirm `v[216:223]` is a valid VGPR range (it is; the
   manual lists 512 VGPRs/SGPRs available on gfx950).
3. The agent should always **cite the manual section** in its
   justification ("per CDNA4 ISA §8.2.3, this load takes ~300 cycles to
   complete"). If it can't, treat the diff with skepticism.

The terminology primer (`01_terminology_primer.md`) will get you to the
point where step 2 takes 30 seconds, not 30 minutes.

---

## 2. What you already have on disk

```
~/ggemm-asm-fwd/                              persistent-fwd-asm-optimization branch
  README.md                                   the bench + optimization summary
  kernels/
    persistent_gemm_ref.co                    Triton-compiled reference
    combined_v5.s   combined_v5.co            optimized FWD/DGRAD (+2.4-3.5%)
    roundtrip.s                               byte-identical reassembly of ref
    variable_k_*  dot_scaled_*                wgrad variants (also here for cross-ref)
    agents/                                   8 parallel-agent variants
  co_compare.cpp                              the C/HIP benchmark
  tools/disasm_to_asm.py  tools/patch_co.py   the assembly toolchain

~/ggemm-asm-wgrad/                            wgrad-asm-optimization branch
  similar layout, focused on wgrad kernel

~/asm-kernel-learning/                        ← this folder
  00_onramp.md          ← you are here
  01_terminology_primer.md
  02_benchmark_walkthrough.md
  03_roundtrip_workplan.md
  refs/cdna4-isa.pdf  refs/cdna3-isa.pdf      AMD's ISA manuals
```

---

## 3. The recommended sequence today

| Step | Time | What | Outcome |
|---|---|---|---|
| 0 | 5 min | Read this doc | Know which doc to open next |
| 1 | 30 min | Read `01_terminology_primer.md` | You can read a 10-line `.s` snippet and explain it in words |
| 2 | 15 min | Read `02_benchmark_walkthrough.md` | You know what "+10%" actually measures |
| 3 | 2-4 hrs | Work through `03_roundtrip_workplan.md` | You've successfully edited a `.s`, re-patched a `.co`, and benchmarked it — `cos=1.000000` |
| 4 | (later) | Read CDNA4 ISA chapters 1-3 on a plane | You stop being intimidated by the manual |

If anything in steps 0-3 takes 2x the estimated time, **stop and ask** before
continuing. Spinning in step 1 for half a day means the primer needs more
context, not that you need to push harder.
