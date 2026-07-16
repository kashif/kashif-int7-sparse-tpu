# Int7+1 Sparse TPU — Engineering Report

**Date:** 2026-07-15 (K=8 revision; original 2026-07-14)
**Target:** Tiny Tapeout TTSKY26c, 2×2 tile, SkyWater SKY130A
**Top module:** `tt_um_kashif_int7_sparse_tpu`
**Clock:** 5 MHz (SPI SCLK <= clk/6)
**Result:** 7 cocotb tests passing (RTL + GL); K=6 baseline measured green
at 58.7% GPL / 50.4% effective on 2×2; K=8 adds ~57 flops

---

## 1. Objective

Implement Roune's "Int7+1" concept: a 7-bit integer multiplier with
1:2 structured sparsity baked into the data format. The 8th bit that
a regular INT8 weight would use for precision is repurposed as a
sparsity select bit — giving 50% sparsity for free, with no pruning
step and no separate sparsity index.

## 2. Design Metrics

| Metric | Value |
|--------|-------|
| Array size | 3×3 = 9 PEs |
| Weight format | Int7+1: {select, value[6:0]}, 8-bit |
| Activation format | INT4 pairs {a_odd, a_even}, 8-bit |
| Multiplier | 7×4-bit signed + 4-bit 2:1 mux |
| Sparsity | 50% baked in (1:2 structured) |
| FFs per PE | 30 (8-bit act pipe + 8-bit weight pipe + 14-bit acc) |
| Gates per PE | ~180-230 |
| Total PEs | 9 |
| Effective MACs/cycle | 18 (9 PEs × 2 contraction steps) |
| Accumulator | 14-bit signed (exact: sparse max 2048, dense max 4096) |
| Contraction depth K | 8 (4 pair steps, each advancing 2 steps); int8 dense K=4 |
| Computation | True GEMM: C = A(3×8) × W(8×3) = C(3×3) |
| Tile size | 2×2 |
| I/O protocol | SPI (16-bit instructions, LSB-first) |
| On-chip memory | Yes (weights + activations, reused across RUNs) |

## 3. Architecture

### Dataflow: Output-Stationary Systolic Wavefront

Activations flow right through horizontal pipes; weights flow down
through vertical pipes; results accumulate in place. Proper skewed
wavefront: row `i` becomes active during counter cycles `[i+1, i+4]`.

```
         a_in[0]   a_in[1]   a_in[2]
            │         │         │
         ┌─────┐   ┌─────┐   ┌─────┐
  W[0]   │ PE  │──►│ PE  │──►│ PE  │
         └──┬──┘   └──┬──┘   └──┬──┘
            │         │         │
         ┌─────┐   ┌─────┐   ┌─────┐
  W[1]   │ PE  │──►│ PE  │──►│ PE  │
         └──┬──┘   └──┬──┘   └──┬──┘
            │         │         │
         ┌─────┐   ┌─────┐   ┌─────┐
  W[2]   │ PE  │──►│ PE  │──►│ PE  │
         └─────┘   └─────┘   └─────┘
```

This is a true matrix-matrix multiply with contraction over K=8.
K was deepened from the inherited reference schedule (3 element steps,
K=6) to 4 steps: +57 flops buys 33% more MACs per RUN and a
power-of-two K that tiles real model dimensions with no padding. The
accumulator widened 13 -> 14 bits because int8 dense mode at K=4 hits
max |C| = 4×128×8 = 4096, one count past 13-bit range.

### PE Design

Each PE receives an activation pair `{a_odd, a_even}` and an Int7+1
weight byte `{select, value[6:0]}`. The select bit muxes which
activation gets multiplied:

```
acc += value × (select ? a_odd : a_even)
```

A 4-bit 2:1 mux replaces a second multiplier — every cycle advances
**two contraction steps** (2× throughput per multiplier).

### Int7+1 Weight Encoding

```
bit[7]    = select: 0 = even position (k=2j), 1 = odd (k=2j+1)
bits[6:0] = 7-bit signed integer value (-64 to +63)
```

12 weight bytes encode a dense-equivalent 8×3 matrix (2× storage and
bandwidth savings). The select bit is consumed combinationally inside
the PE — no `is_odd` port needed, no hierarchy propagation issues.

### Control Unit

Ported from the reference Mini-TPU `control.v`, widened to 16-bit
instructions. Wavefront counter counts 1..8 (last product forms at
r+c+1+j = 8). Accumulator clear on RUN-issue cycle. STORE latches
selection for stable output between SPI transactions.

## 4. Verification

### Test Suite (7 tests, all passing)

| Test | What it verifies |
|------|-----------------|
| `test_known_matmul` | Hand-checked 3×8 × 8×3 matmul |
| `test_select_bit_semantics` | Select bit picks k=2j (sel=0) or k=2j+1 (sel=1) |
| `test_not_degenerate` | Equal-sum inputs give different results (regression) |
| `test_run_clears_accumulators` | Double RUN doesn't double results |
| `test_random` | 12 seeded random trials (3 for GL) |
| `test_dense_mode` | Int7 dense via select=0 (Roune's dense format) |
| `test_dense_int8_mode` | Native int8 dense incl. -128/127, garbage slots ignored, per-RUN mode |

### Golden Model

`golden_matmul()` is deliberately **independent** from the RTL:
decodes Int7+1 bytes into a dense 8×3 matrix, then does a plain
Python `C = A @ W`. Shares no structure with the RTL — cannot pass
artificially by mirroring implementation quirks.

## 5. HDL Guide Compliance

Per [tinytapeout.com/hdl/important/](https://tinytapeout.com/hdl/important/):

- ✅ Top module named `tt_um_kashif_int7_sparse_tpu` (unique, with username)
- ✅ Exact module port definition matching TT template
- ✅ No `initial` blocks; all flops have async reset
- ✅ All outputs assigned (`uo_out`, `uio_out`, `uio_oe`)
- ✅ `(* keep *)` FFs for unused output pins (LVS safety)
- ✅ `_unused` wire for unused inputs
- ✅ `default_nettype none`
- ✅ No `config.tcl` modifications

## 6. Design Rationale (from Roune's document)

Key principles from Roune's "Designing AI Chip Software and Hardware" (2026):

- **Int7+1 format proposed by Roune himself** (Structured sparsity section):
  "I would heavily investigate the possibility of 1:2 sparsity coupled with
  7 bit integer arithmetic. This allows a particularly simple and appealing
  data format: an 8 bit format where the first bit indicates the position of
  the non-zero entry (out of the next two entries) and the remaining 7 bits
  are the bits of that integer entry."

- **"This simple Int7+1 sparse format could, potentially, be the end-state
  for AI inference numerics for many years to come"** (Structured sparsity
  section): Strong validation of the design direction.

- **Dense mode** (Structured sparsity section): "The dense format would just
  set the upper 8th bit to zero." Our design supports this — setting
  `select=0` for all weights gives dense operation at half throughput. Roune
  also suggests optionally supporting INT8 for dense: "you might choose to
  support int8 for the dense format" since "int7 can do pretty much any model
  int8 can, if quantization is done properly."

- **Sparsity for weights only** (Structured sparsity section): "sparsity works
  well for weights, it is much less attractive for activations." Our design
  correctly applies the select bit only to weights; activations are always
  dense INT4.

- **"8 bits is always sufficient for inference"** (Numerics section): Our
  7-bit weights + 4-bit activations are well within the sufficient range.

- **Element-exact MXFP6 (E2M3) for free**: E2M3 values map exactly to 7-bit
  signed integers in the x8 domain (max |60| < 64) — the same fixed-point
  pre-alignment FPGA tensor blocks use for MXFP (arXiv:2607.13898). The host
  converts E2M3 weights to Int7 values and applies E8M0 block scales to the
  exact partial sums; no RTL involvement.

- **Parametrize vector width** (Mono-sized arrays section): "parametrize also
  your hardware design on the vector width... Testing and debugging a 2-wide
  chip is easier than debugging a 256-wide chip." Our array uses `N=3` as a
  constant — could be parametrized for FPGA testing at smaller sizes.

- **"Not all systolic arrays are made equal"** (Numerics section): The 1:2
  sparsity gives 2× throughput per multiplier — "a big decrease in
  power-per-operation" with "only a very modest increase in chip area."

## 7. Known Limitations

- **SPI-bound**: instruction transfer dominates each matmul; K=8 improves
  MACs per instruction ~12-14% over K=6 but the bottleneck remains SPI
- **SCLK domain crossing unsynchronized**: metastability risk if SCLK > clk/6
- **No ReLU by design**: activation functions are host-side (only correct
  after cross-tile accumulation and bias)

## 8. File Structure

```
src/
  project.v              # Top-level TT module
  pe.v                   # Processing element: Int7+1 weight + INT4 MAC
  array.v                # 3×3 output-stationary systolic grid
  control.v              # Instruction-decoded + wavefront counter
  tpu.v                  # TPU top (array + control + memories + output mux)
  spi.v                  # SPI slave (16-bit instructions)
  memory_a.v             # Activation memory (3×8 INT4)
  memory_b.v             # Weight memory (3×4 Int7+1 bytes)
docs/
  info.md                # Datasheet (protocol, comparison table)
  REPORT.md              # This file
  Architecture.drawio    # Architecture diagram
  Dataflow.drawio        # Dataflow diagram
test/
  tb.v                   # Verilog testbench
  test.py                # 7 cocotb tests
  Makefile               # icarus/cocotb build
info.yaml                # TT metadata: 2x2 tile, 5MHz, SKY130A
```

## 9. References

- [Roune, "Designing AI Chip Software and Hardware" (2026)](https://docs.google.com/document/d/1dZ3vF8GE8_gx6tl52sOaUVEPq0ybmai1xvu3uk89_is/edit) — Section "Structured sparsity for systolic arrays": "I would heavily investigate the possibility of 1:2 sparsity coupled with 7 bit integer arithmetic. This allows a particularly simple and appealing data format: an 8 bit format where the first bit indicates the position of the non-zero entry (out of the next two entries) and the remaining 7 bits are the bits of that integer entry."
- [NVIDIA 2:4 Structured Sparsity in Ampere (blog)](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/) — Training recipe: train dense → prune to pattern → retrain with masked weights. Same recipe applies to 1:2.
- [NVIDIA 2:4 paper (Mishra et al. 2021, arXiv:2104.08378)](https://arxiv.org/abs/2104.08378)
- [PFW TPU](https://github.com/wangantian/pfw_tpu) — INT8 2×2 systolic, TT SKY26b
- [Mini-TPU v2](https://github.com/MILOODIAS/IEEE_ttsky_mini_tpu_spi) — INT4 3×3 systolic, TT SKY26b
- [TT HDL Guide](https://tinytapeout.com/hdl/) — FPGA-to-ASIC considerations
- [TT Tech Specs](https://tinytapeout.com/specs/) — Clock, GPIO, memory constraints
- [Companion design: NVFP4 Ternary TPU](https://github.com/kashif/kashif-fp4-ternary-tpu)
