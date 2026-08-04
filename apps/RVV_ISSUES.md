# RVV correctness issues found across `apps/` — analysis and fixes

Scope: `vggnet`, `conv_layer32`, `fc_layer32`, `conv_layer16only`, `fc_layer16only`.

Five distinct defect classes were found, plus a set of build/infrastructure
problems that hid them. Every fix below was verified by a Spike run; unless a
section says otherwise, the observable output was **unchanged**, because most
instances were latent rather than live.

| # | Issue | Live anywhere? | Sites fixed |
|---|-------|----------------|-------------|
| 1 | `vsetvli zero, zero` = keep-`vl`, not VLMAX | yes (2 apps) | 23 |
| 2 | conv `dW` computation commented out | yes (vggnet) | 1 app |
| 3 | `matrix_multiply*` overwrite-vs-accumulate | yes (vggnet) | 3 functions |
| 4 | dx/`col2img` assume a pre-padded layout | yes (vggnet) | 3 apps |
| 5 | batch-mean `1/B` replicated per layer (+2 defects it exposed) | yes (vggnet) | 5 sites |

---

## Issue 1 — `vsetvli zero, zero` does not select VLMAX

### The rule

`vsetvli rd, rs1, vtypei` chooses its AVL from the **register encodings**, not
from any value:

| `rd` | `rs1` | effect |
|------|-------|--------|
| ≠ x0 | ≠ x0  | `vl = min(x[rs1], VLMAX)` — the normal strip-mine form |
| ≠ x0 | **x0** | AVL = ~0 → **`vl = VLMAX`** |
| **x0** | **x0** | **keep the current `vl`**, change only `vtype` |

So `vsetvli zero, zero, e32, m8, ta, ma` means *"keep whatever `vl` is already
set"*. Throughout the tree it was written with the comment `// set vl = VLMAX`.
That misconception is the entire bug: the affected code inherits `vl` from
whatever ran before it, possibly in an unrelated function.

### Failure mode A — splat, then drain (memset-style helpers)

```c
vsetvli zero, zero, e32, m8, ta, ma   // keeps stale vl, e.g. 128
vmv.v.i v16, 0                        // zeroes ONLY 128 lanes; 128..255 are tail
while (n) {
    vsetvli vl, n, e32, m8            // vl = min(n, VLMAX) = 256
    vse32.v v16, (dst)                // stores 256 lanes = 128 zeros + 128 GARBAGE
}
```

The splat is narrower than the stores that consume it. With `ta` (tail-agnostic)
the lanes above the stale `vl` are undisturbed-or-ones — **not zero**.

### Failure mode B — accumulate, then reduce

```c
// ... loop accumulating lane-wise into v8; final iteration has vl = tail ...
vsetvli zero, zero, e32, m8, tu, ma   // keeps the TAIL vl, e.g. 192
vfredsum.vs v0, v8, v0                // reduces only 192 of 256 lanes
```

A reduction reads only `vl` elements of `vs2`, so every accumulator lane above
the tail is silently discarded and the sum comes out too small. Bites whenever
the reduced length is not a multiple of VLMAX.

### Correct form

```c
size_t vl;
asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(vl));       // vl = VLMAX
asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n)); // vl = min(n, VLMAX)
```

Both make correctness independent of the caller. The second is tighter (exactly
the lanes that will be used, so no wasted work); the tree currently uses the
VLMAX form, which is correct but does more work than strictly necessary — see
*Performance* below.

### Sites fixed (23)

| app | file | count | mode |
|-----|------|-------|------|
| vggnet | `kernel/utils.c` | 3 | A |
| vggnet | `kernel/convolution_layer.c` | 3 | A, B |
| vggnet | `kernel/batchnorm_layer.c` | 1 | A |
| conv_layer32 | `kernel/convolution_layer.c` | 4 | 2×A, 2×B |
| conv_layer32 | `kernel/train.c` | 2 | A, B |
| conv_layer32 | `main.c` | 1 | A |
| fc_layer32 | `kernel/train.c` | 1 | B |
| conv_layer16only | (same three files) | 7 | as above |
| fc_layer16only | `kernel/train.c` | 1 | B |

`grep -rn "vsetvli zero, zero"` now matches only explanatory comments.

### Which instances were actually live

Determined by **measuring on target** (reading `vl` via `csrr` and printing the
buffer length), not by reading code — the local source cannot tell you what `vl`
a caller left behind.

* **vggnet, conv zero-padding (mode A) — LIVE.** This was the original bug. It
  corrupted `conv_fwd_pad_scratch`, so conv1's output *changed with batchsize*
  for the same image and weights (image 0 sum −467.53 at bs=4 vs −374.33 at
  bs=32, against a native reference of −466.76 at both). Fixing it brought
  conv1 to −466.76016 and epoch-1 loss to 5.6963 vs native 5.6962.
* **conv_layer32 `zero_f32` in `main.c` (mode A) — LIVE.** Measured incoming
  `vl` = 128 with n = 4096, so `zero_f32(net->conv1.output, 4096)` wrote 2048
  zeros and **2048 lanes of garbage**. Harmless in practice only because its
  sole caller zeroes that buffer immediately before a forward pass that
  overwrites every element.
* **Everything else — dormant**, for shape-dependent reasons:
  * fc_layer32 MSE reduction: `total_elems` = 512 = 2 × VLMAX → no tail.
  * conv_layer32 MSE reduction: `total_elems` = 4096 = 16 × VLMAX → no tail.
  * conv_layer32 `d_bias`: `owoh` = 128 ≤ VLMAX = 256, and the incoming `vl`
    happened to be 256 ≥ 128, so the splat covered every lane the loop touched.
  * conv/fc `16only`: VLMAX doubles to 512 at e16; the same lengths remain
    clean multiples.

> **Correction to an earlier assessment.** An initial static audit called the
> two MSE reductions and `d_bias` "actually wrong today" and treated the splats
> as speculative. Measurement inverted that: all the reductions were fine, and
> the only real defect was in the group with the least local evidence. The
> dormancy of a mode-A site cannot be reasoned about locally at all — it depends
> entirely on unrelated upstream code.

### Performance

The VLMAX form makes some ops execute at more lanes than before — in
`conv_layer32`'s `d_bias`, three ops per output channel go from `vl` = 128 to
256. Spike will not show this (its cycle counter is instruction-shaped and does
not model `vl`), but on Ara/RTL vector op latency scales with `vl`; rough
estimate ~6k extra cycles for that loop. Everything else is unchanged. Using the
`AVL = n` form instead would restore the original cost exactly while keeping the
correctness. Not applied — the `d_bias` cost was deemed unimportant.

---

## Issue 2 — conv weight gradient is commented out

In `conv_layer32/kernel/convolution_layer.c` (and its `16only` twin) both halves
of the `dW` computation are disabled:

* the im2col that builds `x_col`
* the GEMM that fills `t_d_weights` from it

`t_d_weights` is `memset` to zero just above, so the scatter below accumulates
**zeros** into `op->d_weights`; the conv filters receive no gradient. Only
`d_bias` is live, so the bias is the only thing that learns — visible as an
almost flat loss (`0.771228 → 0.771226` over 6 epochs).

**This is deliberate in the benchmark apps** and documented in their comments:
they measure the conv kernel against the FP16 sibling, whose im2col is disabled
because its `vluxei32.v` gather at e16/m4 hangs on Ara RTL. Leaving FP32's
enabled charged it ~2600 cycles FP16 was not paying. **Left as-is.**

**It became a real defect in vggnet**, which inherited the file and actually
trains its conv stack. Symptom: epoch 1 matched the scalar reference exactly,
then epochs 2–4 drifted; a gradient trace showed `conv2/3/4.dW` at literally
`0.000000` where native had real values.

**Fix (vggnet only):** re-enabled im2col + GEMM, feeding `matrix_multiply_32`
from a padding-aware im2col. The gather-offset table is rebuilt per layer,
because it is a single global and vggnet has four conv layers with different
geometry.

---

## Issue 3 — `matrix_multiply*` had two contracts

`matrix_multiply_32` had **three exit paths with two different semantics**,
selected by matrix *shape*:

| path | when | was |
|------|------|-----|
| `matrix_multiply_scalar` fallback | N/K/`padded_m` over the FMATMUL limits | accumulates |
| `fmatmul_32(c, …)` | `padded_m == M` | overwrites |
| M-padding copy-back (`vfadd.vv`) | `padded_m != M` | accumulates |

Its own doc comment said `c = a * b`. A caller that did not pre-zero `c` would
silently sum every invocation onto the previous one — and *which* behaviour you
got depended on `M`, `N`, `K` via `fmatmul_row_block()` and the `FMATMUL_MAX_*`
limits.

**Unified on overwrite** (matching the documentation and `fmatmul_32`):

* scalar fallback now zeroes `c` first,
* the M-padding copy-back is a plain copy instead of load-add-store.

Applied to `fc_layer32/kernel/matrix.c`, `vggnet/kernel/matrix_vec.c`, and
`fc_layer16only/kernel/matrix.c`.

**fp16 was worse than fp32 here.** `matrix_multiply_nt_32` already overwrote
consistently, but `matrix_multiply_nt_16` routed both paths through the scratch
and shared one *accumulating* copy-back — the same function name with opposite
semantics in the two trees, which matters because `fc_layer_test` links kernels
from both into one binary to compare them. Both now overwrite.

Where it was live: vggnet's conv `dW` loop takes the padding branch (`M = oc` is
12 → 16 or 24 → 32). Without a memset it piled all 32 batch images on top of one
another — observed as `conv4.dW` absum 4954 against a native 249. With the
contract fixed, the now-redundant memset was removed from that loop.

`conv_layer32`/`conv_layer16only` also contain `matrix_multiply_nt` with the
accumulating copy-back, reachable only through `fc_layer.c`'s backward, which
these conv-only benchmarks never execute. Left alone — they agree with each
other, which is the parity goal.

---

## Issue 4 — dx and `col2img` assume a pre-padded layout

Two memory conventions exist for the same convolution:

* **pre-padded** — inputs stored already padded, `in_w == out_w + 2`,
  `padding == 0`. What `alexnet_train()` switches `conv1` to in the benchmarks.
* **SAME / unpadded** — `in_w == out_w`, `padding == (k-1)/2`. The logical shape
  `setup_alexnet()` leaves in place, and what any network chaining
  `conv → bn → conv` (i.e. vggnet) actually stores.

`conv_can_use_3x3_dx()` accepted only the first. Under the second it returned 0
and the backward fell through to `col2img`, which is *itself* written for the
pre-padded layout: it indexes `st_x + i` up to `out_w + 1` with a row stride of
`in_w == out_w`, scattering off the end of every row and past the plane. The
vectorized dx path was no better — it re-padded the result into a buffer sized
`in_units`, overflowing by `in_channels*((out+2)² − out²)` floats per image
(conv4: 24·18·18 = 7776 into 6144, i.e. 1632 floats).

**Fixes (all three conv-bearing apps):**

1. `conv_can_use_3x3_dx()` accepts both layouts.
2. `pad` derivation falls back to `(kernel_size-1)/2` when `in_w == out_w`
   (the old `(in_w - out_w)/2` yields 0 and tripped a `pad <= 0` error exit);
   `padded_plane` computed from the actual padded d_output.
3. dx write-back copies straight through under SAME, pads under pre-padded.
4. `col2img` is centred-padding aware and drops out-of-range taps. Under
   pre-padded (`padding == 0`) it reduces exactly to the old expression.

In vggnet this was **live** — it was corrupting `d_input` and cascading into
`conv3.dW`/`conv2.dW`. In the benchmark apps it is latent: their active geometry
is pre-padded and self-consistent.

---

## Issue 5 — the batch-mean factor was replicated per layer

Not a bug in itself — the original arrangement was self-consistent — but a
design that made the code hard to audit and that hid two real defects.

### The old arrangement

The loss is the **mean** over the batch, but `cross_entropy_loss` handed back a
**sum**-convention gradient (`softmax - onehot`, no `1/B`). Each layer then
applied its own `1/B` to its own parameter gradients, in five separate places:

| gradient | where the `1/B` lived |
|---|---|
| conv d_weights | `convolution_layer.c`, trailing vector pass |
| conv d_bias | `convolution_layer.c` |
| bn d_gamma / d_beta | `batchnorm_layer.c`, trailing vector pass |
| fc d_bias | `calc_bias_gradient_vec_32` |
| **fc d_weights** | **inside `fmatmul_tn_32`** |

The last one is the trap: a function named `fmatmul_tn_32` — a general
transposed matmul — silently divided its result by `M`. Nothing at the call
site (`matrix_multiply_tn_32` → `fc_layer.c`) suggests it. `fmatmul_cmp`
benchmarks that same primitive as if it were a plain matmul.

### The change

Apply `1/B` **once**, in `cross_entropy_loss`, and let the chain rule carry it.
Every layer then just sums. Cheaper too: the scale now touches `B × 100 = 3200`
values instead of ~350K (fc1's dW alone is 1536×192), removing several full
load/mul/store sweeps from the backward path.

`fmatmul_tn_32` is now a pure `c = aᵀb`. **Only vggnet's copy changed** —
`fc_layer32`, `fc_layer16only` and `fmatmul_cmp` carry their own copies of that
file and keep the old behaviour, so their numerics are untouched.

Careful: BN's `factor = 1/(batchsize*spatial_size)` in the *forward* is the
batch statistic, and `inv_M` in the backward is part of ∂BN/∂x. Neither is
batch-mean scaling; neither was removed.

### Two defects this exposed

**(a) A scale-sensitive threshold inside the GEMM** — `matrix.c`:

```c
if (apart<0.00001 && apart>(0-0.00001)) //masking for vector processing
```

An **absolute** 1e-5 cutoff on `a`, which in the backward GEMMs *is*
`d_output`. Tuned for the old sum-convention scale; once gradients were 32×
smaller everything in `[1e-5, 3.2e-4)` was silently dropped, and the skip
re-applies at every conv layer — so the error grew with depth (conv4 0.03% →
bn1.dg 6%). Fixed by parameterising it as `matrix_multiply_eps(..., eps)`;
gradient call sites pass `1e-5/batchsize`, forward keeps `1e-5`.

This threshold is a lossy approximation the RVV `fmatmul` path does **not**
have, so it remains a source of scalar-vs-vector divergence independent of
this issue.

**(b) BN's `dx` reused the parameter gradients** — vectorized
`batch_norm_op_backward_full` computes

```c
float factor2 = op->d_beta[c]  * inv_spatial;
float factor3 = op->d_gamma[c] * inv_spatial;
```

Those arrays had already been divided by batchsize, making the product
`Σ/(B·spatial) = Σ/M`. Removing the per-layer scaling left `Σ/spatial` — a
factor of `B` too large — and because this layer's `dx` feeds the next layer
down, it **compounded once per BN**: `conv1.dW` reached 4.5e8. Fixed by
dividing by `M` directly, which also removes the coupling.

The scalar reference does **not** have this coupling (it accumulates into
independent `S1`/`S2` locals), so the reference verified clean and gave a false
all-clear. Only the on-target A/B caught it. `batch_norm_op_backward_input_only`
already used `inv_M` and was correct.

---

## Build / infrastructure problems that hid these

These are not RVV bugs but they repeatedly produced misleading evidence.

**Stale objects.** Several apps declared only `bin/<app>%: Makefile`, which is
the *link* step. Changing a knob like `FC_LAYER32_BATCHSIZE` relinked stale
objects still compiled with the old `-D`, so the binary kept reporting the
previous batch size — this caused at least two wrong conclusions during
debugging. Fixed by adding explicit object lists for `fc_layer32` and
`conv_layer16only`:

```make
FC_LAYER32_SRCS := $(call find_app_sources,fc_layer32)
$(addsuffix .o.spike, $(FC_LAYER32_SRCS)): $(APPS_DIR)/Makefile
```

A recipe-less pattern rule (`app/%.c.o.spike: Makefile`) does **not** work here —
it adds no prerequisite to the pattern rule in `runtime.mk` that actually builds
the object, and make silently ignores it.

**fc_layer32 read past its dataset.** `generated_data/` held 4 samples while
`FC_LAYER32_BATCHSIZE` defaulted to 12, and the batch fetch copies
`batchsize * FC_INPUT_UNITS` floats regardless of `FC_TOTAL_SAMPLES`. It read
1536 floats out of a 512-float array — off the end of `inputs.bin` into
`targets.bin`/`labels.bin` — producing an `inf` loss on the very first forward,
before any weight update. Note `FC_TOTAL_SAMPLES` was *not* the mechanism: it
only bounds the offset, and defaults to the batchsize so it can never detect the
mismatch.

Fixed by generating the data from the same knobs that size the run, via a stamp
file so it regenerates only when the parameters change. Verified: the
regenerated 4-sample blobs are byte-identical to those previously committed, and
at batchsize 12 the loss is now finite and decreasing.

**conv_layer16only did not build at all.** Its `generated_data/` was empty, and
it had **no Makefile block**: `conv_layer`'s rules are scoped `bin/conv_layer.%`
and `conv_layer32`'s `bin/conv_layer32.%`, so it matched neither and was built
with no `-DCONV1_*` defines (falling back to 43/43/8/8 header defaults) and no
`-Wa,-I`, so `.incbin "generated_data/..."` could not resolve. Its committed
`kernel/weights.c` was also internally inconsistent — `conv1_weights[18416]`
matches no valid config (64·16·9 = 9216, 64·32·9 = 18432) while its own
`conv1_bias[16]` implies C_OUT = 16.

Given a proper block (own dim knobs defaulting to `C_IN=64 C_OUT=16 H=1 W=128`,
`-Wa,-I`, `AUTOVECTORIZE=0`, `train.c -O0`, object deps) and regenerated data,
it builds and runs.

---

## vggnet-specific fixes (beyond the five issues)

* **BatchNorm compile error** — `xnorm_ptr`/`dout_ptr` re-declared in the same
  scope in the vectorized backward.
* **BatchNorm byte-vs-element counts** — ten `memset/memcpy_vectorized_f32`
  calls passed `channels * sizeof(float)` where the API takes an *element*
  count, a 4× overwrite that stomped neighbouring BN buffers.
* **Conv forward padding** — vggnet stores feature maps unpadded, so a
  `pad_channels()` step was added before `fconv3d`, and `img2col` plus the
  gather-offset table were made padding-aware so both conv paths consume the
  identical layout and cannot drift apart in convention again.

* **Max pooling** — vectorized forward (strided 2×2 max, argmax encoded with a
  `vmerge` chain) and backward (`vmseq.vi` + masked `vsse32.v` scatter).
  The tie-break **must** match the scalar reference: that version scans
  x-outer/y-inner with a strict `>`, so the first maximum wins, giving priority
  P0 > P2 > P1 > P3. `vmerge` keeps the *last* match, so the merges are applied
  lowest-priority-first. `max` has no unique subgradient at a tie, so this is
  not a correctness bug — but it is what keeps the build reproducing the
  reference, which is the comparison that validates everything else.
* **ReLU** — `vfmax.vf` forward, masked select (`vmfgt.vf` + `vmerge.vvm`)
  backward, as one flat sweep over `batchsize*units`. `vfmax` matches
  `in > 0 ? in : 0` exactly for `-0.0` and NaN; the select also avoids the
  `inf*0 → NaN` the scalar multiply-by-mask form could produce.

### Verification

Three independent instruments, in increasing order of strength.

**1. Per-layer checksums.** A one-shot `sum`/`absum`/`max` of the first forward
and first backward on both sides, walking the chain until the first line
disagrees. `-DDUMP_TRACE` in `vggnet/main.c` and `kernel/train.c`, flag-gated.
This found Issues 1–4.

**2. Element-wise audit** (`mp_audit_indices`, `maxpooling_layer.c`, under
`DUMP_TRACE`). Recomputes the scalar argmax and checks both the stored index and
the scattered gradient value for every output:

```
[MPAUDIT] units=49152  ties=11595  index_mismatch=0  value_mismatch=0
[MPAUDIT] units=98304  ties=36150  index_mismatch=0  value_mismatch=0
```

147,456 outputs containing 47,745 exact ties, zero mismatches — the vectorized
max-pool backward reproduces the scalar gradient routing *and* values exactly.
This is strictly better than the loss comparison: it isolates one kernel, gives
a yes/no answer rather than a judgement call, and needs only a single run, so
there are no cross-run confounds.

**3. Convention A/B.** For Issue 5, build old and new conventions and diff the
backward traces. Expected invariant: parameter gradients **identical**,
activation gradients **exactly 32× smaller**.

| | reference (`~/vggnet`, `-O0`) | vectorized (Spike/RVV) |
|---|---|---|
| parameter gradients | bit-identical | bit-identical |
| activation gradients | exactly 32× (rel ≤ 3e-8) | exactly 32× (rel ≤ 5e-8) |
| 4-epoch trajectory | identical to 4 dp | — |

This is what caught defects (a) and (b) above; neither was visible in the loss.

**Cross-tree trajectory** (`FINETUNE=0`, batch 32, 1 step/epoch):

| epoch | reference (scalar) | Spike (vectorized) | rel. diff |
|-------|--------------------|--------------------|-----------|
| 1 | 6.1565 | 6.1564 | 1.6e-5 |
| 2 | 6.2500 | 6.2522 | 3.5e-4 |
| 3 | 6.2976 | 6.2643 | 5.3e-3 |
| 4 | 5.9383 | 5.9427 | 7.4e-4 |

Epoch 1 is the only checkpoint before a weight update compounds, and it agrees
to 1e-5. Beyond it, **end-to-end loss is a poor oracle**: training is a chaotic
map, the surviving differences are fp reduction ordering (vector tree reductions
and `-ffast-math` on Spike vs sequential scalar sums), and any code change
reshuffles the last bits. Divergence at epoch 4 is not evidence of a bug, and
agreement there is not evidence of correctness — prefer instruments 2 and 3.

---

## vggnet16 — FP16 port, and the recurrence of Issues 2 & 4

`vggnet16/` is a pure-FP16 (`_Float16`) copy of vggnet: every weight, activation,
gradient and velocity is FP16, matmul/conv accumulate in FP16 (no widening), and
there are **no FP32 master weights** — momentum is the only accumulator that
carries a sub-epsilon update across steps. Only two things stay FP32: the
softmax/cross-entropy `exp` (there is no FP16 transcendental), whose gradient is
narrowed back to FP16 with `vfncvt.f.f.w`; and BN's `1/(B·spatial)` divisors,
which are ∂-math, not master weights.

The compute core was **reused** from the proven single-layer FP16 apps
(`fmatmul*` and `matrix.c` from `fc_layer16only`; `fconv3c` + `convolution_layer.c`
from `conv_layer16only`), renamed `_16 → _32` so vggnet's call sites are
unchanged. That reuse is exactly what re-imported two bugs this file already
documents for FP32 — because the single-layer apps predate those fixes.

### Recurrence A (Issue 4) — the conv forward never padded its input

`conv_op_forward_3x3_ara` (from `conv_layer16only`) fed the **unpadded**
activation straight into `fconv3d_CHx3x3_f16`, which indexes a 3×3 window
assuming rows are `out_w + 2` wide. The single-layer app got away with it (its
test input was pre-padded); the deep net stores activations unpadded, so every
conv read off the end of each row. Inner batch images read into the next image's
data (bounded); the **last** image read past the buffer into stale memory.

FP16 made this loud where FP32 would have hidden it: conv1 for image 3 hit
**41888 → bn1 = inf → every downstream layer 0 → uniform logits → loss = ln(100)
exactly**. The fix is the same `pad_channels` into `conv_fwd_pad_scratch` that
FP32 vggnet already does. After it, the FP16 forward tracks FP32 layer-for-layer
(conv1 2.490 vs 2.489, fc1 10.16 vs 10.18).

> Lesson restated: an OOB read that FP32's range silently absorbs, FP16 turns
> into inf. The precision downgrade is a *bug detector*, not just a hazard.

### Recurrence B (Issue 2) — conv dW GEMM still commented out

`conv_layer16only` deliberately left the `matrix_multiply` dW GEMM disabled (it
benchmarked the forward kernel), so `t_d_weights` stayed zeroed and the scatter
wrote zeros. In vggnet16 that meant **all four `conv*.dW = 0`** — the conv
filters never trained (biases and BN did). Re-enabled exactly as in FP32:
`precompute`-free scalar `img2col` on the padded image (`conv_pad_input_image`)
feeding `matrix_multiply_32`. `img2col` was also made padding-aware
(`in_w_p = in_w + 2·padding`) — re-enabling it without that would just have
reintroduced Recurrence A in the backward.

### Verification (the FP16-specific instruments)

1. **Forward checksums** vs FP32 — match layer-for-layer after the padding fix.
2. **Isolated conv unit test** (`VGGNET16_SELFTEST=1`, `conv_selftest()` in
   `convolution_layer.c`): a fixed 3×3 SAME conv fed inputs that are all `k/8`
   — *exactly* representable in FP16, so identical to the FP32 inputs. Result:
   forward, `dx` **and** `dW` are **bit-identical** between the FP16 and FP32
   builds, including a **permutation-sensitive checksum** `Σ pᵢ·(i+1)`. A
   `col2img` that routed a gradient to the wrong pixel would change that
   checksum; it doesn't. This is the clean isolated test end-to-end loss can't
   be: identical inputs, no rounding (values stay in FP16's exact range), so
   agreement is only possible if the *indexing* is identical.
3. **No silent inf.** `bw_trace` accumulates in FP32, so a printed `absum` of
   337,877 is a sum over 5184 elements, not one value; the max element was 356.5
   (< 65504). An inf anywhere would print `absum` as 2147483647.

### FP16 training machinery (from `fc_layer16only`, one scale for the whole net)

* **Dynamic loss scaling**, applied once: `CLEAR_FFLAGS` before the loss, seed
  scaled by `loss_scale/B` in FP32 then narrowed to FP16, gradients flow scaled
  through the whole backward, `READ_FFLAGS` after. Overflow → halve scale and
  **skip the optimizer** that step; 8 consecutive underflows or 1000 clean steps
  → grow it; clamped to `[1, 32768]`.
* **`sgd_step_vec`** unscales by `1/loss_scale` as its first op, then momentum +
  weight-decay + `[-1,1]` clip — all FP16, no master copy.
* `SET_FRM_RNE` for reproducibility.

Every kernel file was `-fsyntax-only`-checked against the real sysroot as it was
converted, which is why the full link surfaced only one issue (a `0f16`/`1f16`
invalid-literal in the generated `weights.c` — dropped the suffix).

### FP16-specific known-open

* Only `FINETUNE=0` with the small (64-image) embedded set was exercised. The
  loss just bounces in the FP32 regime — too few updates on too little data to
  show learning in *either* precision. Whether FP16 *learns* comparably to FP32
  over real training (full dataset, many steps) is unmeasured.
* The dynamic loss scale never left 1024 in these runs, so the halve/grow and
  skip-step branches are **present but unexercised on target**.
* Not done (premise disproven by the trace — the ln(100) collapse was the
  padding bug, not near-zero init or underflow): a selectable He-init gain and
  an `INPUT_SCALE` headroom knob. Both remain reasonable *features*, neither is
  a *fix*.

---

## Known-open

* `conv_layer16only` reports `loss: 0.000000` where `conv_layer32` reports
  `0.771228`. Predates all changes here; likely fp16 underflow in the loss or
  its extraction path. **Not investigated.**
* `fc_layer32` prints `loss: 2147483647.000000` (`INT32_MAX`, i.e. `printf_`
  rendering inf) in some configurations. Pre-existing.
* vggnet's finetune path (`FINETUNE=1`) has not been re-run since the conv
  backward rework or the Issue 5 convention change. It freezes conv so it never
  executed the broken code, but that is reasoning, not evidence — and Issue 5(b)
  was exactly a defect that reasoning-from-the-reference declared absent.
  `batch_norm_op_backward_input_only` and `fc_op_backward_input_only_32` look
  correct by inspection (they use their own `S1`/`S2` and `inv_M`) but are
  **unverified on target**.
* The 1e-5 magnitude skip in `matrix_multiply` (Issue 5a) has no counterpart in
  the RVV `fmatmul` path, so the scalar and vector GEMMs are not numerically
  equivalent by construction. Currently small in effect; not quantified.
* `matrix_vec.c`'s scalar fallbacks (`matrix_multiply_scalar*`) still carry the
  unparameterised 1e-5 cutoff. They are dormant (workspace-overflow paths only,
  and they announce themselves with `[SCALAR]`), so they were left alone — but
  if one ever fires under the new convention it will silently drop gradients.
* Legacy `vfredsum` mnemonic (pre-RVV-1.0) is still mixed with `vfredusum`.
  Which ordering it aliases to is assembler-dependent.
* `fc_layer32/generated_data/.params` is a new build artifact and wants a
  `.gitignore` entry.
