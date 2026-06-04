# Terminology Primer: Reading a GPU `.s` File

> **Purpose:** Every term in the vendor's READMEs (`s_setprio`, `buffer_load
> hoisting`, `vmcnt`, MFMA, occupancy, `ds_write drain`, etc.) explained from
> first principles, with concrete pointers into the `.s` files you have.
>
> **You should be able to:** after reading this, look at any 10-line slice of
> `~/ggemm-asm-fwd/kernels/combined_v5.s` and explain in plain English what
> each instruction is doing and why.

---

## 1. The mental model: how a GPU executes a kernel

A GPU is a stack of **compute units (CUs)** — MI355X has 256 of them. Each
CU has:

- **VGPRs** (vector general-purpose registers): 512 of them on gfx950. Each
  VGPR is a 32-bit value held *per thread*. A wavefront has 64 threads, so
  one VGPR is actually 64 × 32 bits = 256 bytes of physical SRAM.
- **SGPRs** (scalar general-purpose registers): 512 of them. One value
  shared across the whole wavefront. Used for loop counters, addresses,
  flags — anything that's the same across threads.
- **LDS** (Local Data Share): 160 KiB on gfx950, fast SRAM shared within a
  workgroup. NVIDIA calls this "shared memory."
- **HBM** (High-Bandwidth Memory): the main global memory, slow (hundreds
  of cycles to read), high-throughput.

A **kernel** is a program that runs on every CU in parallel. Each CU runs
some number of **wavefronts** simultaneously — that count is the kernel's
**occupancy**. Occupancy 1 means one wavefront per CU at a time. Higher
occupancy = more parallelism = better latency hiding.

### The optimization economy

99% of GPU optimization is one sentence: **the MFMA (matrix multiply) units
are starving for data, and you're trying to feed them faster than they can
consume.**

To do that you must:

1. Issue HBM loads (slow) early enough that the data is ready when MFMA
   needs it.
2. Stage data through LDS (fast, but limited) without making LDS the new
   bottleneck.
3. Keep the wavefront issuing instructions, not stalling on `s_waitcnt`.
4. Use the densest MFMA opcode the hardware supports (more FLOPs/instruction).

The vendor's optimizations in `~/ggemm-asm-fwd/README.md` and
`~/ggemm-asm-wgrad/README.md` are all instances of (1), (2), and (3). The
Triton compiler picks the opcode for (4).

---

## 2. The instruction families you'll see

Look at the first 50 lines of `~/ggemm-asm-fwd/kernels/combined_v5.s`. You
see prefixes like `s_`, `v_`, `buffer_`, `ds_`. Here's the cheat sheet:

| Prefix | Family | Meaning |
|---|---|---|
| `s_` | Scalar | One operation across the whole wavefront. Uses SGPRs. Cheap. Examples: `s_load_dwordx2`, `s_branch`, `s_waitcnt`, `s_barrier`. |
| `v_` | Vector | Per-thread operation. Uses VGPRs. The bulk of compute. Examples: `v_mov_b32`, `v_add_u32`, `v_mfma_*`. |
| `buffer_` | Global memory | Loads/stores to HBM (slow, ~300+ cycles). Examples: `buffer_load_dwordx4`, `buffer_store_short`. |
| `ds_` | LDS | Loads/stores to shared memory (fast, ~20 cycles). Examples: `ds_read_b128`, `ds_write_b128`. |
| `flat_` | Generic memory | Address-decoded: goes to HBM or LDS depending on the pointer. Slower than `buffer_` when you know it's HBM. |

### MFMA — the workhorse instruction

`v_mfma_f32_16x16x128_f8f6f4 acc[0:3], v[0:3], v[4:7], acc[0:3]` does a
**16×16 outer product over 128 K-elements** in fp8, accumulating into 4
fp32 VGPRs (a 16×16 tile of fp32 outputs per thread group). One MFMA
issues every 32 cycles on gfx950 and does ~16,000 FLOPs per instruction.

- **gfx950 FP8 MFMA opcodes you'll see:**
  - `v_mfma_f32_16x16x128_f8f6f4` (forward / dgrad in `combined_v5.s`) —
    16,000 FLOPs/instr, 16×16 tile.
  - `v_mfma_f32_32x32x64_f8f6f4` (dot_scaled wgrad in `dot_scaled_v2.s`) —
    131,000 FLOPs/instr, 32×32 tile. **8× the throughput** of the legacy
    `16x16x32_fp8_bf8` instruction.
  - `v_mfma_f32_16x16x32_fp8_bf8` (legacy wgrad in `variable_k_wgrad_mega.s`)
    — 16,000 FLOPs/instr, the older fp8 MFMA from CDNA3.

If you ever see a `v_mfma_*` and want to know its throughput, look it up in
the CDNA4 ISA PDF's MFMA opcode table. The "K" in the name is the inner
reduction dimension consumed per instruction; FLOPs = 2 × M × N × K.

### `buffer_load_*` — the latency you're hiding

`buffer_load_dwordx4 v[0:3], v100, s[8:11], 0` loads 16 bytes (4 dwords)
from HBM at the address computed from `v100` + `s[8:11]` into VGPRs
`v[0:3]`. **The instruction doesn't wait for the data.** It issues, and the
data arrives in `v[0:3]` ~300+ cycles later. You wait for it with
`s_waitcnt vmcnt(N)` (see below).

This decoupling is the entire reason "buffer_load hoisting" works:

```asm
buffer_load_dwordx4 v[0:3], ...        ; issue load. Data NOT ready yet.
v_mfma_f32_*  ...  ...  ...            ; do 50 MFMAs. Compiler can choose
v_mfma_f32_*  ...  ...  ...            ;   to overlap them with the load.
v_mfma_f32_*  ...  ...  ...
... (~300 cycles of MFMA later) ...
s_waitcnt vmcnt(0)                     ; now stall until the load finishes
v_mfma_f32_*  v[0:3], ...              ; use the loaded data
```

If you instead issue the load right before the `s_waitcnt`, you spend ~300
cycles stalled doing nothing. That's the "+10%" win in the wgrad kernel:
moving the load earlier in the program text changes when it executes
relative to MFMAs.

### `ds_read` / `ds_write` — the LDS staging

LDS is the bridge between HBM and the MFMA registers. A common pattern:

```
buffer_load HBM → VGPR  (slow, async)
ds_write    VGPR → LDS  (fast, async; needs lgkmcnt to wait)
ds_read     LDS → VGPR  (fast, async)
v_mfma      VGPR → VGPR  (compute)
```

Double-buffering ping-pongs two LDS regions so the MFMAs always have data
in flight. The "ds_write drain" in `combined_v5.s` is the sequence of 8
`ds_write`s that flush the current buffer to LDS between K-iterations.

### `s_waitcnt vmcnt(N)` / `lgkmcnt(N)` — the synchronization primitives

The hardware tracks in-flight memory operations with counters:

- **`vmcnt`** — number of outstanding VMEM (HBM) loads/stores not yet
  complete. `s_waitcnt vmcnt(0)` means "stall until all outstanding HBM
  ops are done." `s_waitcnt vmcnt(3)` means "stall until at most 3 are
  outstanding" — used to drain in **FIFO order** when you've issued more
  loads than you want hanging.
- **`lgkmcnt`** — same idea but for LDS, GDS, KM (scalar mem). `s_waitcnt
  lgkmcnt(0)` waits for all LDS reads/writes to complete.
- **`expcnt`** — export (output to graphics; not relevant here).

These are the most-frequent instructions in the `.s` file. **Removing a
redundant `s_waitcnt` is one of the cheapest possible optimizations** (one
fewer stall point) and is exactly what bullet 2 of the dot_scaled wgrad
optimizations does.

### `s_barrier` — the workgroup sync

Forces all wavefronts in a workgroup to reach this point before any
continues. Expensive. The combined_v5 kernel has 8 barriers per K-loop
iteration; the README cites them as a 14% structural overhead.

### `s_setprio 0..3` — the wave-priority hint

`s_setprio 3` tells the CU's instruction scheduler "this wave has higher
priority than other waves on this CU; prefer to schedule its instructions."
`s_setprio 0` is "normal/low priority."

In `combined_v5.s` the pattern is `s_setprio 3` right after `s_barrier`
(start of the MFMA-heavy section) and `s_setprio 0` right before
`buffer_load` (yield while waiting for HBM). Net effect: the MFMA wave
gets favored when there's work to do, and other waves get to run during
memory stalls. **rocBLAS uses the same trick** — it's not magic, it's a
scheduler hint.

This is only relevant when occupancy > 1 (there's competition for the CU).
At occupancy 1 it's nearly free but doesn't help either, which is why the
README shows "+0.1-0.3%" and notes "no benefit at occupancy=2" in the
opposite direction depending on context.

---

## 3. The optimization techniques, decoded

### "Buffer_load hoisting"

Take a `buffer_load_dwordx4` instruction and move it **earlier in the
program text**, so the load issues sooner and has more time to complete
before its result is needed. Constraints:

1. The destination VGPRs must be **dead** (not holding live data) at the
   new location. The README explicitly identifies the dead-VGPR set:
   `v[238:245]` in fwd, `v[216:223]` in wgrad legacy.
2. The address registers used by the load (e.g. `v194`, `v193` for fwd)
   must hold valid values at the new location.
3. The vmcnt-wait at the consumer must still drain the load FIFO in the
   right order (loads complete in issue order; `vmcnt(3)` drains 3 oldest
   loads first).

When the README says "extending HBM latency cover from ~120 to ~3500
cycles," that's "before hoisting the load issues 120 cycles before its
use; after hoisting, 3500 cycles before its use." 3500 cycles is comfortably
more than the ~300-cycle HBM round-trip, so the load fully overlaps with
compute and the `s_waitcnt vmcnt(0)` becomes a no-op.

### "MFMA / ds_write interleaving with drain"

Default Triton output groups all `ds_write`s back-to-back, then runs all
remaining MFMAs:

```
ds_write B0   ds_write B1   ...   ds_write B7
MFMA #1   MFMA #2   ...   MFMA #4
```

But each `ds_write` requires draining a vmcnt-wait (the buffer was loaded
from HBM, written to LDS). The drain waits are stall cycles. If the
tail MFMAs are **independent of the `ds_write` source registers** (i.e.
they read different VGPRs), you can move them into the stall windows:

```
MFMA #1
ds_write B0   ds_write B1     ← these drain during MFMA #1's 32-cycle execution
MFMA #2
ds_write B2   ds_write B3
... etc
```

The 4 tail MFMAs go from costing 4 × ~30-cycle stall to ~0 (they overlap
the drain). The +0.6-0.8% is that recovered stall.

### "Redundant MOV / waitcnt elimination"

Triton emits things like `v_add_u32_e32 v200, 0, v201` (add 0 to v201,
store in v200) — that's just a copy. If you can show the destination
register isn't read between the copy and the next overwrite, the copy is
dead and can be removed. Same for back-to-back `s_waitcnt lgkmcnt(0)`
with no new LDS ops between them — the second one waits on nothing.

Each removal saves one instruction issue slot, ~1 cycle. Many small wins
compound (the wgrad kernel got +0.3% from removing 6 MOVs).

### "ds_read overlap"

ds_read has ~20-cycle latency. If two ds_reads share a destination VGPR
("ds_read v[10:11]; use v[10:11]; ds_read v[10:11]; use v[10:11]"), you
can't issue the second until the first's use is done. But if you have a
**dead pair of VGPRs available** (the wgrad README mentions `v[138:139]`
which were holding scale factors no longer needed), you can ping-pong:

```
ds_read v[10:11]
ds_read v[138:139]    ← issue immediately, alternative destination
use v[10:11]          ← first ds_read result ready ~20 cycles later
use v[138:139]        ← second ds_read result ready ~20 cycles after that
```

Both ds_reads are in flight simultaneously, hiding their latency behind
each other.

### "Loop tail restructure"

The very last MFMA of the K-loop can co-execute with the loop control
instructions (`s_add_i32` to bump the counter, `s_cbranch` to loop back).
If you reorder so the loop control sits in the MFMA's 32-cycle issue
window, you save the loop-control cycles entirely.

---

## 4. How to read the `.s` file

Look at `~/ggemm-asm-fwd/kernels/combined_v5.s` line 8 onwards:

```asm
s_load_dwordx2 s[2:3], s[0:1], 0x0       ; load 2 scalars (lhs ptr) from kernarg
s_load_dwordx8 s[4:11], s[0:1], 0x8      ; load 8 scalars (rhs ptr, out ptr, etc)
s_load_dwordx4 s[12:15], s[0:1], 0x28    ; load 4 more scalars
s_waitcnt lgkmcnt(0)                     ; wait for all scalar loads to finish
s_branch .L0                              ; jump to label .L0
```

Translation: "Read my kernel arguments (pointers and sizes) from the
constant buffer at `s[0:1]`, wait for them to arrive, then jump past the
NOP-padded function prologue to the real code."

The `s_nop 0` block at lines 13-68 is **mandatory hardware-required NOP
padding** (gfx950 has specific alignment rules for code following a
branch). The README confirms: "NOP removal: 0% — All are hardware-required
hazard NOPs."

---

## 5. Workflow when reviewing an agent's diff

When an agent proposes a change, your review checklist is:

1. **What instruction did it move/add/remove?** Look it up in the CDNA4 ISA
   PDF (search for the mnemonic). Confirm the agent's claim about its
   semantics.
2. **Are the registers it touches dead?** If the agent claims `v[216:223]`
   is dead at the new location, search for `v216` through `v223` in the
   `.s` file and confirm those VGPRs aren't live across the change.
3. **Does the waitcnt ordering still hold?** If the agent moves a
   `buffer_load`, every `s_waitcnt vmcnt(N)` between old and new positions
   has its FIFO meaning changed. The agent should explicitly verify this.
4. **What's the bench number?** Run `co_compare --benchmark --warmup 50
   --iters 200`. Demand `cos >= 0.999999` for correctness. Demand a
   measurable speedup (>0.3%) that holds across 5 reruns.
5. **What's the cost?** Does the change extend a VGPR's live range and
   risk hurting occupancy? Did it add any new instructions? (Sometimes
   adding 4 instructions to save 200 cycles of stall is a win; sometimes
   it tips occupancy from 1 to 0.5 and is a 10% regression.)

If the agent can't answer (1)-(3) in its own words, with citations, it's
guessing. Don't merge.
