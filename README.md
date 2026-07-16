![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# Int7+1 Sparse Mini-TPU

A 3x3 output-stationary systolic array with **1:2 structured sparsity baked
into the weight format** — Roune's Int7+1 concept: a 7-bit integer multiplier
where the 8th bit encodes which of two adjacent contraction steps holds the
non-zero value. Plus a **native int8 dense mode** at half throughput, so
off-the-shelf int8-quantized models map directly.

- [Read the documentation for project](docs/info.md)
- Architecture and dataflow diagrams: [docs/Architecture.drawio](docs/Architecture.drawio), [docs/Dataflow.drawio](docs/Dataflow.drawio)

## The Core Idea

From Roune's "Designing AI Chip Software and Hardware" (2026), Section
"Structured sparsity for systolic arrays":

> "I would heavily investigate the possibility of 1:2 sparsity coupled with
> 7 bit integer arithmetic. This allows a particularly simple and appealing
> data format: an 8 bit format where the first bit indicates the position of
> the non-zero entry (out of the next two entries) and the remaining 7 bits
> are the bits of that integer entry."

### Weight Encoding (8 bits)

```
bit[7]    = select: which of two consecutive contraction steps (k=2j or
            k=2j+1) holds the value; the other step is zero
bits[6:0] = 7-bit signed integer value (-64 to +63)
```

The sparsity runs along the **contraction axis k** (as in NVIDIA 2:4), so in
hardware the select bit steers a 4-bit mux in front of a single 7x4
multiplier:

```
acc += value * (select ? a_odd : a_even)
```

**A mux replaces a second multiplier.** Every cycle advances two contraction
steps: 2x throughput per multiplier, 2x weight storage/bandwidth (12 bytes
encode a dense-equivalent 8x3 matrix), zero sparsity metadata — unlike 2:4,
which needs a separate index.

### Two dense modes

- **Int7 dense**: set all select bits to 0 (Roune: "the dense format would
  just set the upper 8th bit to zero") — half throughput.
- **Native int8 dense** (RUN flag): each byte is a full int8 weight for one
  contraction step (K=4). Same PEs, same 8-cycle wavefront — the sparse/dense
  2x throughput gap is measurable on one chip.

### Bonus: element-exact MXFP6 (E2M3)

Every MXFP6 E2M3 value maps exactly onto a 7-bit signed integer in the x8
domain (subnormals 1-7, then 8+M scaled by 2^(E-1); max |60| < 64), so the
chip runs **MXFP6 (E2M3) weights natively** — the accuracy-preferred MXFP6
variant. The host converts E2M3 nibbles+sign to Int7 values, feeds them
through either the 1:2-sparse or int8-dense path, and applies the E8M0
per-32 block scales to the exact partial sums during dequantization (as the
companion FP4 chips do for NVFP4/MXFP4). This is the same fixed-point
pre-alignment used by FPGA tensor blocks for MXFP
([arXiv:2607.13898](https://arxiv.org/abs/2607.13898)): E2M1 fits 5 signed
bits, E2M3 fits exactly 7 — the Int7 container.

## Architecture

Output-stationary systolic array following the silicon-proven
[Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi):
activation pairs flow right, Int7+1 weight bytes flow down, results
accumulate in place. Computes `C = A(3x8 INT4) x W(8x3)` exactly in 14-bit
accumulators, in an 8-cycle skewed wavefront. K=8 (sparse) tiles the
power-of-two layer widths of real models with no padding.

```
            W col 0    W col 1    W col 2      (Int7+1 bytes, skewed)
               |          |          |
A row 0 --> [PE 00] -> [PE 01] -> [PE 02]      A rows: INT4 pairs
               |          |          |          (2 contraction steps
A row 1 --> [PE 10] -> [PE 11] -> [PE 12]       per cycle)
               |          |          |
A row 2 --> [PE 20] -> [PE 21] -> [PE 22]
```

### SPI instruction set (16 bits, LSB-first; SCLK <= clk/6)

| Instruction | Format (binary)        | Description |
|-------------|------------------------|-------------|
| `LOAD A`    | `10 0 rr eee 0000aaaa` | INT4 activation into row `r` (0-2), element `e` (0-7) |
| `LOAD B`    | `10 1 cc 0ss wwwwwwww` | Weight byte into column `c` (0-2), pair slot `s` (0-3) |
| `RUN`       | `01 d 0000000000000`   | Clear accumulators, run 8 cycles; `d`=1 selects int8 dense |
| `STORE`     | `11 b rr cc 000000000` | Result byte of C[r][c] on `uo_out` (`b`: low byte / high 6 bits) |

Pins: `ui[0]`=MOSI, `ui[1]`=CS, `ui[2]`=SCLK; `uo_out`=result byte;
`uio[0]`=MISO, `uio[1]`=ready.

## File Structure

```
src/
  project.v     # Top-level TT module (tt_um_kashif_int7_sparse_tpu)
  tpu.v         # Core: control + memories + array + result mux
  spi.v         # SPI slave, 16-bit instructions, 126-bit readback
  control.v     # LOAD/RUN/STORE decode, skewed wavefront counter
  memory_a.v    # Activations: 3 rows x 8 INT4, read as pairs
  memory_b.v    # Weights: 3 cols x 4 Int7+1 bytes (= dense 8x3)
  array.v       # 3x3 systolic array, 14-bit accumulators
  pe.v          # Int7+1 sparse / int8 dense MAC
test/
  tb.v          # Verilog testbench (GL_TEST compatible)
  test.py       # 7 cocotb tests with independent golden model
  Makefile      # icarus/cocotb build
info.yaml       # TT metadata: 2x2 tile, 5 MHz, SKY130A
```

## Verification

7 cocotb tests drive the SPI interface like an external host and compare
against an **independent golden model** (Int7+1 decode to dense matrix, plain
matmul — shares no structure with the RTL):

| Test | Description |
|------|-------------|
| `test_known_matmul` | Hand-checked matmul, mixed select bits |
| `test_select_bit_semantics` | Select bit picks k=2j vs k=2j+1, verified per row |
| `test_not_degenerate` | Equal-sum activation matrices must differ (guards against w*sum collapse) |
| `test_run_clears_accumulators` | Back-to-back RUNs don't double results |
| `test_random` | 12 randomized full-coverage trials |
| `test_dense_mode` | Int7 dense via select=0 (Roune's dense format) |
| `test_dense_int8_mode` | Native int8 dense incl. -128/127, garbage in ignored slots, mode switching |

CI: RTL tests, GDS build, TT precheck, and gate-level test all green
(K=6 baseline measured 58.7% utilization on 2x2; K=8 adds ~57 flops).

## ASIC Notes (per TT HDL guide + reference REPORT lessons)

- All flops async-reset (`dfrtp`) — no gate-level X-poisoning from no-reset
  pipeline registers
- `(* keep *)` FFs drive `uio_oe`/`uio_out` — avoids conb/VGND LVS merges
- Fixed uio directions; host-driven pins never output-enabled
- 14-bit accumulators: exact for both modes (sparse max 2048, dense max 4096)
- `default_nettype none`, no `initial` blocks, `_unused` wire

## Target

- **Shuttle**: TTSKY26c (SkyWater SKY130A)
- **Tile**: 2x2 (~167x108 um each)
- **Clock**: 5 MHz (SPI SCLK <= 833 kHz)

## References

- [Roune, "Designing AI Chip Software and Hardware" (2026)](https://docs.google.com/document/d/1dZ3vF8GE8_gx6tl52sOaUVEPq0ybmai1xvu3uk89_is/edit) — Section "Structured sparsity for systolic arrays"
- [NVIDIA 2:4 Structured Sparsity in Ampere (blog)](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/) — 2:4 needs a separate index; Int7+1's select bit IS the index
- [Jack of All Scales: A Versatile FPGA Tensor Block for MXFP Precisions (arXiv:2607.13898)](https://arxiv.org/abs/2607.13898) — MXFP-to-fixed-point pre-alignment; E2M3 -> 7 signed bits (the MXFP6 compatibility above)
- [Mini-TPU v2](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi) — INT4 3x3 systolic, TT SKY26b; architecture and SPI protocol base
- [PFW TPU](https://github.com/wangantian/pfw_tpu) — INT8 2x2 systolic, TT SKY26b
- [TT HDL Guide](https://tinytapeout.com/hdl/) / [TT Tech Specs](https://tinytapeout.com/specs/)
- [Companion design: NVFP4 Ternary TPU](https://github.com/kashif/kashif-fp4-ternary-tpu)

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that aims to make it easier and cheaper than ever to get your digital and analog designs manufactured on a real chip. To learn more and get started, visit https://tinytapeout.com.
