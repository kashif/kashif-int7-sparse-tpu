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
steps** — 2x throughput per multiplier, and 9 weight bytes encode a
dense-equivalent **6x3** weight matrix (2x storage and bandwidth).

The design computes `C = A x W` with `A` a 3x6 INT4 activation matrix and `W`
the 6x3 dense-equivalent weight matrix. Results are exact 12-bit signed values
(no truncation). Architecture, SPI protocol, and skewed-wavefront control
follow the proven reference mini-TPU
([MILOUDIAS/IEEE_ttsky_mini_tpu_spi](https://github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi)),
widened to a 16-bit instruction. Block and dataflow diagrams:
`docs/Architecture.drawio`, `docs/Dataflow.drawio`.

### Instruction set (16 bits, sent LSB-first over SPI)

| Instruction | Format (binary)        | Description |
|-------------|------------------------|-------------|
| `LOAD A`    | `10 0 rr eee 0000aaaa` | INT4 activation `a` into row `r` (0-2), element `e` (0-5) |
| `LOAD B`    | `10 1 cc 0ss wwwwwwww` | Int7+1 byte `w` into column `c` (0-2), pair slot `s` (0-2) |
| `RUN`       | `01 00000000000000`    | Clear accumulators, run the wavefront (7 cycles) |
| `STORE`     | `11 b rr cc 000000000` | Drive result byte of C[r][c] on `uo_out`: `b`=0 low byte, `b`=1 high nibble |

SCLK must be at most clk/6 (the SPI bit counter crosses clock domains
unsynchronised, as in the reference). The `ready` pin (uio[1]) pulses when a
RUN completes; alternatively just wait 7+ clock cycles.

## How to test

Run the cocotb testbench:

```
cd test
make -B
```

The suite drives the SPI interface exactly like an external host and checks
the full `C = A x W` result against an **independent golden model** (Int7+1
bytes decoded to a dense 6x3 matrix, then a plain matrix multiply). It
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
| Computation | Matrix-matrix (GEMM) | Matrix-vector | **True GEMM (K=6)** |
| PE registers | 3 | 2 | **3** |
| PE gates | ~200+ | ~74 | **~180-230** |
| FFs per PE | 12 | 14 | **28** |
| Effective MACs/cycle | 9 | 16 | **18 (2× sparsity)** |
| I/O | SPI (12-bit instr) | Direct pin streaming | **SPI (16-bit instr)** |
| On-chip memory | Yes (weights + acts) | No | **Yes (weights + acts)** |
| Weight reuse | Yes | No | **Yes** |
| ReLU | No | Yes | **No** |
| Tile | 1×1 | 1×1 | **1×2** |
| Accumulator | 4-bit (mod 16) | 10-bit signed | **12-bit signed (exact)** |
| Numerics | Educational INT4 | NVFP4 (Blackwell) | **Int7+1 (Roune's concept)** |

### Key Innovation: Sparsity for Free

From Roune's talk ("Numerics: The Unsung Competitive Battleground"):

> "A 7-bit integer multiplier with 1:2 structured sparsity baked in.
> The 8th bit repurposed to encode which of two adjacent entries is
> non-zero. You get sparsity for free in the data format."

Unlike NVIDIA's 2:4 structured sparsity (arXiv:2104.08378) which
requires a separate index, Int7+1 encodes the sparsity pattern in the
data itself — no extra metadata, no pruning step needed.
