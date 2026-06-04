# Round-Trip Work Plan: Your First Hands-On Today

> **Goal:** By the end of this exercise (~2-4 hours), you will have:
>
> 1. A working `co_compare` binary on your dev box.
> 2. A `.s` file you disassembled from a `.co` and reassembled, producing a
>    bit-identical kernel (`cos = 1.000000`, `max_diff = 0.000000`).
> 3. The same `.s` with one trivial, intentional edit (a renamed comment or
>    reordered NOP) that still produces bit-identical output — proving you
>    can edit and the toolchain re-applies your changes.
> 4. Confidence that you can hand a `.s` to an AI agent and trust the
>    bench loop to catch any mistake it makes.
>
> **What this does NOT cover today:** Actually optimizing the kernel.
> That's the next exercise. Today is purely about the **edit → assemble →
> benchmark loop** working reliably on your machine.
>
> **Prereqs:** You already have `~/ggemm-asm-fwd` and `~/ggemm-asm-wgrad`
> checked out. A previous agent already reproduced the +1.16× wgrad number,
> so HIP + the GPU are working. Today is even simpler.

---

## Phase 0 — Sanity (~10 min)

### 0.1 Confirm you're on the MI355X dev box and ROCm works

```bash
hostname -s
rocm-smi --showproductname | head -5
which hipcc
hipcc --version
which llvm-mc || ls /opt/rocm/llvm/bin/llvm-mc
```

**Expected:**
- `hipcc` reports ROCm 6.x or 7.x.
- `rocm-smi` shows an MI355X (gfx950) device.
- `llvm-mc` exists at `/opt/rocm/llvm/bin/llvm-mc` (the standard location).

**If anything's missing:** stop and ask. Do not improvise installs.

### 0.2 Build the bench binary

```bash
cd ~/ggemm-asm-fwd
hipcc -O3 -o co_compare co_compare.cpp
ls -l co_compare
```

**Expected:** `co_compare` binary, ~150-300 KB.

### 0.3 Run the bench on the unmodified reference (sanity baseline)

```bash
cd ~/ggemm-asm-fwd
export HIP_VISIBLE_DEVICES=0
./co_compare kernels/persistent_gemm_ref.co kernels/persistent_gemm_ref.co \
    --benchmark --warmup 50 --iters 200 --site gate_up_fwd
```

**Expected output (shape will match):**

```
Device: AMD Instinct MI355X (256 CUs)
  kernel: _grouped_fp8_persistent_gemm_kernel

gate_up_fwd      M=131072 D1=2880 D2=5760  A=378MB B=1062MB Out=1509MB
  [PASS] cos=1.000000  max_diff=0.000000  ref_nz=...  test_nz=...  nan=0
  ref=2.272 ms (1960 TFLOPS)  test=2.272 ms (1960 TFLOPS)  speedup=1.0000x
```

**Sanity checks:**
- `cos=1.000000` and `max_diff=0.000000` — comparing a kernel against
  itself must be bit-identical.
- `speedup=1.0000x` — also identity.
- `ref` and `test` are within ±0.5% of each other (run-to-run noise).
- The TFLOPS roughly matches the README (1960 ± 50).

**If `cos < 1.0`:** something is broken with the bench harness. **Stop.**

### 0.4 Confirm `combined_v5.co` also passes against ref

```bash
./co_compare kernels/persistent_gemm_ref.co kernels/combined_v5.co \
    --benchmark --warmup 50 --iters 200 --site gate_up_fwd
```

**Expected:** `cos=1.000000`, `max_diff=0.000000`, `speedup ≈ 1.024x`
(the README's +2.4%).

This confirms the vendor's optimized kernel is bit-identical to the ref
on these inputs and ~2.4% faster. **If you can't reproduce this number,
stop and investigate before going further.**

---

## Phase 1 — Round-trip the reference kernel (~30 min)

The "round-trip" exercise: take a working `.co`, disassemble it to text,
re-assemble that text back to a `.co`, and confirm the new `.co` is
**byte-identical to the original**. This proves the entire toolchain is
faithful before you start changing anything.

The vendor has already done this for both kernels — see
`~/ggemm-asm-fwd/kernels/roundtrip.s` (their byte-identical reassembly of
`persistent_gemm_ref.co`). Today you'll reproduce that yourself.

### 1.1 Make a working directory

```bash
mkdir -p ~/asm-kernel-learning/exercise01
cd ~/asm-kernel-learning/exercise01
cp ~/ggemm-asm-fwd/kernels/persistent_gemm_ref.co ref.co
ls -la ref.co
```

### 1.2 Disassemble the reference

```bash
/opt/rocm/llvm/bin/llvm-objdump -d ref.co > ref_disasm.txt
wc -l ref_disasm.txt
head -30 ref_disasm.txt
```

**Expected:** A few thousand lines. The header shows ELF info; the body
shows instruction encodings interleaved with mnemonics. This is what
`tools/disasm_to_asm.py` consumes.

### 1.3 Convert to a re-assembleable `.s`

The vendor's tool requires a metadata JSON (extracted from the original
`.co`'s `.note` section). Easiest path: use the vendor's existing
`roundtrip.s` as your starting point — they did the metadata extraction
already. Compare it against your disassembly to confirm they match.

```bash
diff <(grep -v '^$' ref_disasm.txt | head -100) \
     <(grep -v '^$' ~/ggemm-asm-fwd/kernels/roundtrip.s | head -100) \
   | head -50
```

You'll see formatting differences (the `.s` adds labels, the disasm has
raw addresses), but the instruction sequence should match line for line
after labels.

### 1.4 Reassemble and patch

```bash
cd ~/asm-kernel-learning/exercise01
python3 ~/ggemm-asm-fwd/tools/patch_co.py \
    ref.co \
    ~/ggemm-asm-fwd/kernels/roundtrip.s \
    roundtrip.co \
    --llvm-mc /opt/rocm/llvm/bin/llvm-mc
ls -l ref.co roundtrip.co
```

**Expected:**
- Both files exist.
- Their **sizes are identical**.
- `patch_co.py` printed "Original .text: offset=0x..., size=N bytes" and
  "New code: N bytes" with **N equal in both lines** (no padding needed).

### 1.5 Bytewise compare

```bash
cmp ref.co roundtrip.co && echo "BYTE-IDENTICAL" || echo "DIFFER"
md5sum ref.co roundtrip.co
```

**Expected:** `BYTE-IDENTICAL`. Both md5s match.

**If they differ:** Run `cmp -l ref.co roundtrip.co | head -20` to see
where. The most common cause is that `roundtrip.s` was written against a
slightly different ref `.co` than the one in your tree. That's a real
bug to chase — flag it and don't continue until resolved.

### 1.6 Bench the round-trip

```bash
cd ~/ggemm-asm-fwd  # need co_compare and the other .co's in this dir
./co_compare kernels/persistent_gemm_ref.co \
    ~/asm-kernel-learning/exercise01/roundtrip.co \
    --benchmark --warmup 50 --iters 200 --site gate_up_fwd
```

**Expected:** `cos=1.000000`, `max_diff=0.000000`, `speedup` within ±0.5%
of 1.0.

**This is the milestone for Phase 1.** Once you see this, the
edit→assemble→bench loop is provably faithful on your machine.

---

## Phase 2 — Make a trivial edit and confirm bit-identical output (~30 min)

Now you'll edit the `.s` in a way that **cannot possibly affect the math**
and confirm the patched `.co` still produces bit-identical output. This is
the test that proves your edits are surviving the toolchain correctly.

### 2.1 Pick a safe edit

Two trivially-safe edits, in increasing order of interest:

**Option A — Comment change.** Add a comment to `roundtrip.s` (lines
prefixed with `;` are comments and don't affect codegen).

```bash
cp ~/ggemm-asm-fwd/kernels/roundtrip.s ~/asm-kernel-learning/exercise01/edit1.s
# Open edit1.s and add a comment line somewhere inside the function body
```

For example, insert `; my first edit` somewhere after `.L0:` and before
the next instruction.

**Option B — Swap two adjacent NOPs.** The prologue has a long run of
`s_nop 0`s (lines 13-68 of `combined_v5.s`). Swapping any two of them is a
no-op semantically and a no-op binary-wise (they encode identically). It
should still produce a byte-identical `.co` after reassembly.

Pick A for the first run.

### 2.2 Reassemble

```bash
cd ~/asm-kernel-learning/exercise01
python3 ~/ggemm-asm-fwd/tools/patch_co.py \
    ref.co \
    edit1.s \
    edit1.co \
    --llvm-mc /opt/rocm/llvm/bin/llvm-mc
```

**Expected:** `New code` size equals `Original .text` size. (Comments don't
emit bytes; the assembled output is the same.)

```bash
cmp ref.co edit1.co && echo "BYTE-IDENTICAL" || echo "DIFFER"
```

**Expected:** `BYTE-IDENTICAL`. A comment in the `.s` produces the same
machine code as no comment.

### 2.3 Bench the edited kernel

```bash
cd ~/ggemm-asm-fwd
./co_compare kernels/persistent_gemm_ref.co \
    ~/asm-kernel-learning/exercise01/edit1.co \
    --benchmark --warmup 50 --iters 200 --site gate_up_fwd
```

**Expected:** Identical to Phase 1.6's output — `cos=1.000000`, `max_diff=0`,
speedup ≈ 1.0.

---

## Phase 3 — Make a real edit that changes bytes but not behavior (~45 min)

Now graduate to an edit that **does** change the assembled bytes, but is
known semantically safe — so you exercise the diff-vs-bench feedback loop
end to end.

### 3.1 Pick the edit: insert `s_nop 0` somewhere it can do no harm

Add one extra `s_nop 0` at the **very end** of the function, just before
`s_endpgm`. Look at the last 20 lines of `roundtrip.s`:

```bash
tail -20 ~/ggemm-asm-fwd/kernels/roundtrip.s
```

You'll see something like:

```asm
    s_waitcnt vmcnt(0)
    s_endpgm
```

Make a copy and insert one `s_nop 0` between them:

```bash
cp ~/ggemm-asm-fwd/kernels/roundtrip.s ~/asm-kernel-learning/exercise01/edit2.s
# Edit edit2.s manually with your editor of choice. Insert one
#     s_nop 0
# line on its own, immediately before `s_endpgm`.
```

**Why this is safe:** `s_endpgm` terminates the kernel. Anything between
`s_waitcnt vmcnt(0)` and `s_endpgm` runs after all useful work is done.
One extra `s_nop` adds 4 bytes (one instruction word) of latency that
nobody waits on.

### 3.2 Reassemble — and observe that this time, sizes will differ

```bash
cd ~/asm-kernel-learning/exercise01
python3 ~/ggemm-asm-fwd/tools/patch_co.py \
    ref.co \
    edit2.s \
    edit2.co \
    --llvm-mc /opt/rocm/llvm/bin/llvm-mc
```

**Expected:** `patch_co.py` prints "WARNING: size mismatch!" because
you've added 4 bytes. But the patch script handles this by NOP-padding the
shorter side, so the patch will succeed if your new code is ≤ original
size. Since you just *added* an instruction, you'll see:

```
ERROR: new code is larger than original — cannot patch in place
```

That's expected! The reference `.text` has no slack space for added
instructions.

**Resolution:** Instead, *replace* one existing `s_nop 0` with another
`s_nop 0`. Same instruction, same bytes, but it exercises that you can
locate and edit a specific instruction. Pick line 13 of `roundtrip.s`
(the first `s_nop 0` after `s_branch .L0`) and edit it to literally the
same instruction. The `.co` will be bit-identical after reassembly.

For an edit that actually changes bytes but stays within the size budget,
the cleanest exercise is to **remove** a `s_nop` (saves 4 bytes; the
script will NOP-pad the remainder back to original size with `s_nop 0`,
which is what the existing nop you removed was anyway, so still no
behavior change).

### 3.3 Bench it

```bash
cd ~/ggemm-asm-fwd
./co_compare kernels/persistent_gemm_ref.co \
    ~/asm-kernel-learning/exercise01/edit2.co \
    --benchmark --warmup 50 --iters 200 --site gate_up_fwd
```

**Expected:** `cos=1.000000`, `max_diff=0.000000`, speedup within
run-to-run noise of 1.0.

---

## Phase 4 — What you have now (~5 min)

By the end of Phase 3 you've proven:

1. The full ROCm + HIP + llvm-mc toolchain works on your box.
2. You can disassemble a `.co`, edit the `.s`, reassemble back to a `.co`,
   and the kernel still runs correctly.
3. The bench loop catches both correctness (`cos`, `max_diff`) and
   performance changes, and both numbers agree with the README baseline.
4. You understand size-constraint pitfalls (patch_co's "can't grow"
   limit) and know to design edits that swap instructions in place rather
   than insert.

This is the **minimum viable kernel-tuning loop**. Everything from here on
— buffer_load hoisting, MFMA interleaving, `s_setprio` tweaks — is
"compose edits A, B, C and rerun bench." The agent does the edits; you
review them; the bench is the oracle.

---

## Phase 5 — What's next (after today)

Save this for a separate session. Likely sequence:

1. **Reproduce one published optimization yourself.** Pick the simplest:
   `s_setprio` insertion (+0.2%). Apply the diff between `roundtrip.s` and
   `combined_v5.s` for just the `s_setprio` instructions. Bench. You
   should see the README's +0.2%.
2. **Revert an optimization from `combined_v5.s` and watch the bench
   regress.** Pick the buffer_load hoisting (largest, +2.0%). Move the
   hoisted load back to its original position. Bench. Should regress by
   ~2%.
3. **Now you understand the loop deeply enough to brief agents.** Hand a
   single targeted optimization brief to N parallel Claude Code agents
   ("try waitcnt elimination; show me the diff and the bench result").
   Merge winners.

---

## Troubleshooting cheat sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `cmp ref.co roundtrip.co` differs | Local `roundtrip.s` doesn't match local `ref.co` (versions drifted) | Re-extract metadata from your `ref.co` with `llvm-objdump` / `llvm-readobj --notes` and regenerate `.s` from scratch via `tools/disasm_to_asm.py` |
| `patch_co.py`: "new code is larger" | You added instructions; `.text` has no slack | Either replace an `s_nop 0` instead of adding, or drop something equal-sized |
| `cos < 0.999` after edit | You changed program semantics, not just scheduling | Revert; this is the bench's job — trust it |
| TFLOPS very low (< 1000 on gate_up_fwd) | DVFS not warmed up; HIP cache cold; GPU contended | Re-run; verify nothing else is on the GPU (`rocm-smi --showuse`) |
| `hipModuleLoadData` fails | `.co` is corrupt; ELF metadata broken | The patch tool only edits `.text`; if metadata's wrong, `ref.co` is the culprit, not your edit |

---

## When you finish

Append a short note to your daily Confluence log (only if you want — see
the daily-confluence-log rule):

> Completed kernel-tuning toolchain bring-up. Round-tripped
> persistent_gemm_ref.co byte-identical; verified bench loop produces
> cos=1.000000 max_diff=0 on edit/no-edit; matched README's +2.4% on
> combined_v5.co. Ready to drive Claude Code agents on real optimization
> work next session.

Otherwise just save your `~/asm-kernel-learning/exercise01/` directory.
It's your foundation for everything that comes after.
