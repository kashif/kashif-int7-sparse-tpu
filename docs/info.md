<!---
This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

A mini TPU built around a **3x3 output-stationary systolic array** whose weight
format bakes in **1:2 structured sparsity**: the **Int7+1** format. Each 8-bit
weight byte `{select, value[6:0]}` carries a 7-bit signed value plus a select
bit that says which of **two consecutive contraction steps** (k = 2j or
k = 2j+1) the value occupies — the other position is zero. The 8th bit that a
plain INT8 weight would spend on precision is repurposed as the sparsity
index, in the spirit of NVIDIA's 2:4 sparsity (arXiv:2104.08378) but simpler
and cheaper.

The hardware payoff is in the processing element: the activation pipe carries
an INT4 *pair* (two contraction steps), and the select bit steers a 4-bit
2:1 mux in front of a single 7x4-bit signed multiplier:

```
acc += value * (select ? a_odd : a_even)
```

A mux replaces a second multiplier, so **every cycle advances two contraction
steps** — 2x throughput per multiplier, and 12 weight bytes encode a
dense-equivalent **8x3** weight matrix (2x storage and bandwidth).

The design computes `C = A x W` with `A` a 3x8 INT4 activation matrix and `W`
the 8x3 dense-equivalent weight matrix. Results are exact 14-bit signed values
(no truncation), and K=8 tiles the power-of-two layer widths of real models
with no padding. A **native int8 dense mode** (RUN instruction flag) reuses
the same PEs with each byte as a full int8 weight for one contraction step
(K=4, half throughput) — so off-the-shelf int8-quantized models map directly,
per Roune's compatibility recommendation. Architecture, SPI protocol, and skewed-wavefront control
follow the proven reference mini-TPU
([MILOUDIAS/IEEE_ttsky_mini_tpu_spi](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi)),
widened to a 16-bit instruction. Block and dataflow diagrams:
`docs/Architecture.drawio`, `docs/Dataflow.drawio`.

### Instruction set (16 bits, sent LSB-first over SPI)

| Instruction | Format (binary)        | Description |
|-------------|------------------------|-------------|
| `LOAD A`    | `10 0 rr eee 0000aaaa` | INT4 activation `a` into row `r` (0-2), element `e` (0-7) |
| `LOAD B`    | `10 1 cc 0ss wwwwwwww` | Int7+1 byte `w` into column `c` (0-2), pair slot `s` (0-3) |
| `RUN`       | `01 d 0000000000000`   | Clear accumulators, run the wavefront (8 cycles); `d`=0 Int7+1 sparse (K=8), `d`=1 int8 dense (K=4) |
| `STORE`     | `11 b rr cc 000000000` | Drive result byte of C[r][c] on `uo_out`: `b`=0 low byte, `b`=1 high 6 bits |

SCLK must be at most clk/6 (the SPI bit counter crosses clock domains
unsynchronised, as in the reference). The `ready` pin (uio[1]) pulses when a
RUN completes; alternatively just wait 8+ clock cycles.

## How to test

Run the cocotb testbench:

```
cd test
make -B
```

The suite drives the SPI interface exactly like an external host and checks
the full `C = A x W` result against an **independent golden model** (Int7+1
bytes decoded to a dense 8x3 matrix, then a plain matrix multiply). It
includes select-bit semantics tests, a non-degeneracy test (equal-sum
activation matrices must produce different results), accumulator-clear checks,
and randomized full-coverage trials.

## External hardware

None required. Any SPI-capable host (e.g. the demo board's RP2040) drives
MOSI/CS/SCLK and reads the result bytes on `uo_out`.

---

## Comparison with Other TT TPU Designs

| Feature | Mini-TPU (IEEE) | NVFP4 Ternary TPU | **This design** |
|---------|-----------------|-------------------|-----------------|
| Array | 3×3 = 9 PEs | 4×4 = 16 PEs | **3×3 = 9 PEs** |
| Weight format | INT4 | E2M1 (NVFP4) | **Int7+1 (7-bit + select)** |
| Activation | INT4 | Ternary {-1,0,+1} | **INT4 pairs** |
| Multiplier | 4×4 hardware | None (MUX-add) | **7×4 hardware + 4-bit mux** |
| Sparsity | None | None | **50% baked in (1:2)** |
| Computation | Matrix-matrix (GEMM) | Matrix-vector | **True GEMM (K=8)** |
| PE registers | 3 | 2 | **3** |
| PE gates | ~200+ | ~74 | **~180-230** |
| FFs per PE | 12 | 14 | **28** |
| Effective MACs/cycle | 9 | 16 | **18 (2× sparsity)** |
| I/O | SPI (12-bit instr) | Direct pin streaming | **SPI (16-bit instr)** |
| On-chip memory | Yes (weights + acts) | No | **Yes (weights + acts)** |
| Weight reuse | Yes | No | **Yes** |
| ReLU | No | Yes | **No** |
| Tile | 1×1 | 1×1 | **1×2** |
| Accumulator | 4-bit (mod 16) | 10-bit signed | **14-bit signed (exact)** |
| Native int8 dense mode | No | No | **Yes (RUN flag, half rate)** |
| Numerics | Educational INT4 | NVFP4 (Blackwell) | **Int7+1 (Roune's concept)** |

### Key Innovation: Sparsity for Free

From Roune's "Designing AI Chip Software and Hardware" (2026), Section "Structured sparsity for systolic arrays":

> "I would heavily investigate the possibility of 1:2 sparsity coupled
> with 7 bit integer arithmetic. This allows a particularly simple and
> appealing data format: an 8 bit format where the first bit indicates
> the position of the non-zero entry (out of the next two entries) and
> the remaining 7 bits are the bits of that integer entry."

> "I think this simple Int7+1 sparse format could, potentially, be the
> end-state for AI inference numerics for many years to come."

Unlike NVIDIA's 2:4 structured sparsity (arXiv:2104.08378) which
requires a separate index, Int7+1 encodes the sparsity pattern in the
data itself — no extra metadata, no pruning step needed.

### Dense Modes (two of them)

Per Roune: "The dense format would just set the upper 8th bit to zero."
Setting `select=0` for all weights places values at even positions only
(k=2j), with odd positions always zero. This gives **dense int7 operation
at half throughput** — same as NVIDIA's dense mode for 2:4 sparsity.

Additionally, per Roune's compatibility argument ("many models are quantized
to int8 and not int7... your customers will have an easier time finding
off-the-shelf quantized models if you do support int8"), the RUN
instruction's `d` flag enables **native int8 dense mode**: each weight byte
is a full int8 value for a single contraction step (K=4), and only even
activation slots (elements 0, 2, 4, 6) participate. Same PEs, same 8-cycle
wavefront — sparse Int7+1 gets K=8 in the time int8 dense gets K=4, which
is exactly the 2x structured-sparsity throughput claim, measurable on one
chip.

### Element-Exact MXFP6 (E2M3) Compatibility

Every MXFP6 E2M3 element converts exactly to a 7-bit signed integer in the
x8 integer domain, so E2M3-quantized weights run natively on this chip —
through the 1:2-sparse path (after 1:2 pruning) or the int8 dense path —
with the E8M0 per-32-element block scales applied by the host to the exact
partial sums:

| E2M3 field pattern | Value | x8 integer |
|--------------------|-------|------------|
| E=0 (subnormal), M=1..7 | M/8 | 1..7 |
| E=1, M=0..7 | (8+M)/8 | 8..15 |
| E=2, M=0..7 | (8+M)/4 | 16..30 (even) |
| E=3, M=0..7 | (8+M)/2 | 32..60 (step 4) |

Max magnitude is 60 (< 64), the sign bit negates, and products stay within
the accumulator bounds (|w| <= 60 is stricter than the int8 dense bound).
This is the fixed-point pre-alignment used by FPGA tensor blocks for MXFP
(arXiv:2607.13898): E2M1 needs 5 signed bits, E2M3 exactly 7 — Int7 is the
natural container for the accuracy-preferred MXFP6 variant.

### Sparsity Applies to Weights, Not Activations

Per Roune: "sparsity works well for weights, it is much less attractive
for activations." Our design correctly applies sparsity only to weights
(the select bit is in the weight byte). Activations are always dense INT4.
