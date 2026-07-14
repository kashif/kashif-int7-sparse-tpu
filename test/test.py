# SPDX-FileCopyrightText: (c) 2026 Kashif
# SPDX-License-Identifier: Apache-2.0
#
# Int7+1 sparse mini-TPU tests.
#
# The golden model is an INDEPENDENT dense matrix multiply built from
# first principles (decode Int7+1 bytes into a dense 6x3 matrix, then
# plain C = A @ W) — it shares no structure with the RTL, so it cannot
# "pass artificially" by mirroring implementation quirks.

import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

GL_TEST = bool(os.environ.get("GATES") == "yes")

N = 3      # array is N x N
K = 6      # contraction depth (K/2 = 3 Int7+1 pair slots)

OP_RUN   = 0b01
OP_LOAD  = 0b10
OP_STORE = 0b11

# SPI pin positions within ui_in
PIN_MOSI = 0
PIN_CS   = 1
PIN_SCLK = 2

# SCLK half-period in clk cycles (SCLK = clk/8, well under the clk/6 limit)
SCLK_HALF = 4


# ----------------------------------------------------------------------
# Instruction encoding (16 bits, sent LSB-first)
# ----------------------------------------------------------------------

def instr_load_a(row, elem, value):
    return (OP_LOAD << 14) | (0 << 13) | (row << 11) | (elem << 8) | (value & 0xF)


def instr_load_b(col, slot, byte):
    return (OP_LOAD << 14) | (1 << 13) | (col << 11) | (slot << 8) | (byte & 0xFF)


def instr_run(dense=0):
    return (OP_RUN << 14) | (dense << 13)


def instr_store(row, col, byte_sel):
    return (OP_STORE << 14) | (byte_sel << 13) | (row << 11) | (col << 9)


# ----------------------------------------------------------------------
# Golden model
# ----------------------------------------------------------------------

def int7p1_decode(byte):
    """{select, value[6:0]} -> (select, signed value)."""
    select = (byte >> 7) & 1
    value = byte & 0x7F
    if value & 0x40:
        value -= 0x80
    return select, value


def decode_weights(wbytes):
    """9 Int7+1 bytes (wbytes[col][slot]) -> dense 6x3 signed matrix."""
    W = [[0] * N for _ in range(K)]
    for c in range(N):
        for j in range(K // 2):
            sel, val = int7p1_decode(wbytes[c][j])
            W[2 * j + sel][c] = val
    return W


def golden_matmul(A, wbytes):
    """C = A (3x6 signed INT4) x dense(W) (6x3). Exact — max |C| = 1536."""
    W = decode_weights(wbytes)
    return [[sum(A[i][k] * W[k][c] for k in range(K)) for c in range(N)]
            for i in range(N)]


def golden_dense_int8(A, wbytes):
    """int8 dense mode: each byte is a full int8 weight for ONE step;
    only even activation slots (elements 0, 2, 4) participate (K=3).
    Exact — max |C| = 3 * 128 * 8 = 3072, fits 13-bit signed."""
    def s8(b):
        return b - 256 if b & 0x80 else b
    return [[sum(A[i][2 * j] * s8(wbytes[c][j]) for j in range(K // 2))
             for c in range(N)] for i in range(N)]


# ----------------------------------------------------------------------
# SPI driver
# ----------------------------------------------------------------------

async def spi_send(dut, instr):
    """Bit-bang one 16-bit instruction, LSB-first, sampled on SCLK rising."""
    def drive(mosi, cs, sclk):
        dut.ui_in.value = (mosi << PIN_MOSI) | (cs << PIN_CS) | (sclk << PIN_SCLK)

    drive(0, 0, 0)
    await ClockCycles(dut.clk, SCLK_HALF)
    for i in range(16):
        bit = (instr >> i) & 1
        drive(bit, 0, 0)                      # setup MOSI while SCLK low
        await ClockCycles(dut.clk, SCLK_HALF)
        drive(bit, 0, 1)                      # rising edge samples the bit
        await ClockCycles(dut.clk, SCLK_HALF)
    drive(0, 1, 0)                            # CS high between instructions
    # Leave time for the clk-domain data_ready pulse and execution
    # (a RUN needs 7 cycles; this gap covers it).
    await ClockCycles(dut.clk, 12)


async def hw_reset(dut):
    dut.ena.value = 1
    dut.ui_in.value = 1 << PIN_CS   # CS idle high
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)


# ----------------------------------------------------------------------
# High-level operations
# ----------------------------------------------------------------------

async def load_operands(dut, A, wbytes):
    for i in range(N):
        for k in range(K):
            await spi_send(dut, instr_load_a(i, k, A[i][k]))
    for c in range(N):
        for j in range(K // 2):
            await spi_send(dut, instr_load_b(c, j, wbytes[c][j]))


async def read_result(dut, row, col):
    await spi_send(dut, instr_store(row, col, 0))
    low = int(dut.uo_out.value)
    await spi_send(dut, instr_store(row, col, 1))
    high = int(dut.uo_out.value)
    val = ((high & 0x1F) << 8) | low        # 13-bit signed accumulator
    if val & 0x1000:
        val -= 0x2000
    return val


async def run_matmul(dut, A, wbytes, dense=0):
    await load_operands(dut, A, wbytes)
    await spi_send(dut, instr_run(dense))
    return [[await read_result(dut, i, c) for c in range(N)] for i in range(N)]


def check(dut, got, expected, label):
    for i in range(N):
        for c in range(N):
            assert got[i][c] == expected[i][c], (
                f"{label}: C[{i}][{c}] expected {expected[i][c]}, "
                f"got {got[i][c]} (full: got={got} expected={expected})")


def start_clock(dut):
    cocotb.start_soon(Clock(dut.clk, 100, unit="ns").start())


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

@cocotb.test()
async def test_known_matmul(dut):
    """Hand-checked matmul: distinct weights and activations."""
    start_clock(dut)
    await hw_reset(dut)

    # A[i][k] = small distinct values
    A = [[1, 2, -1, 3, 0, -2],
         [0, 1, 2, -3, 1, 1],
         [-1, -1, 2, 2, -4, 3]]
    # wbytes[col][slot]: mix of select=0 and select=1
    wbytes = [[0x03, 0x85, 0x02],   # col 0: W[0]=3, W[3]=5, W[4]=2
              [0xFE, 0x04, 0x81],   # col 1: W[1]=-2, W[2]=4, W[5]=1
              [0x7F, 0x00, 0xC0]]   # col 2: W[0]=-1, (zero), W[5]=-64

    results = await run_matmul(dut, A, wbytes)
    dut._log.info(f"results: {results}")
    check(dut, results, golden_matmul(A, wbytes), "known")
    dut._log.info("known matmul PASSED")


@cocotb.test()
async def test_select_bit_semantics(dut):
    """The select bit must pick position k=2j (0) or k=2j+1 (1)."""
    start_clock(dut)
    await hw_reset(dut)

    # Activations distinct at every k so misrouting is visible
    A = [[1, 2, 3, 4, 5, 6],
         [7, -8, -7, -6, -5, -4],
         [-3, -2, -1, 1, 2, 3]]

    # Single weight: value 5 in col 0 slot 1 (covers k=2,3)
    for sel in (0, 1):
        wbytes = [[0x00, (sel << 7) | 0x05, 0x00],
                  [0x00, 0x00, 0x00],
                  [0x00, 0x00, 0x00]]
        results = await run_matmul(dut, A, wbytes)
        for i in range(N):
            expected = 5 * A[i][2 + sel]
            assert results[i][0] == expected, (
                f"sel={sel}: C[{i}][0] expected {expected}, got {results[i][0]}")
            assert results[i][1] == 0 and results[i][2] == 0
    dut._log.info("select-bit semantics PASSED")


@cocotb.test()
async def test_not_degenerate(dut):
    """Two activation matrices with identical row sums must give
    different results — guards against the w*sum(acts) failure mode
    the previous design had."""
    start_clock(dut)
    await hw_reset(dut)

    A1 = [[1, 2, 3, 4, 5, 6]] * 3
    A2 = [[6, 5, 4, 3, 2, 1]] * 3          # same row sums, reversed order
    wbytes = [[0x01, 0x02, 0x03],           # W[0]=1, W[2]=2, W[4]=3
              [0x81, 0x82, 0x83],           # W[1]=1, W[3]=2, W[5]=3
              [0x40, 0x00, 0x00]]           # W[0]=-64

    r1 = await run_matmul(dut, A1, wbytes)
    r2 = await run_matmul(dut, A2, wbytes)
    check(dut, r1, golden_matmul(A1, wbytes), "A1")
    check(dut, r2, golden_matmul(A2, wbytes), "A2")
    assert r1 != r2, ("equal-sum inputs gave identical outputs — "
                      "design has collapsed to w*sum(acts) again")
    dut._log.info("non-degeneracy PASSED")


@cocotb.test()
async def test_run_clears_accumulators(dut):
    """Each RUN starts from zero — results must not double on rerun."""
    start_clock(dut)
    await hw_reset(dut)

    A = [[1, 1, 2, 2, 3, 3],
         [4, 4, 5, 5, 6, 6],
         [-1, -2, -3, -4, -5, -6]]
    wbytes = [[0x07, 0x87, 0x07],
              [0x86, 0x06, 0x86],
              [0x05, 0x85, 0x05]]

    expected = golden_matmul(A, wbytes)
    await load_operands(dut, A, wbytes)
    await spi_send(dut, instr_run())
    await spi_send(dut, instr_run())    # second RUN, same operands
    results = [[await read_result(dut, i, c) for c in range(N)]
               for i in range(N)]
    check(dut, results, expected, "rerun")
    dut._log.info("accumulator clear PASSED")


@cocotb.test()
async def test_random(dut):
    """Randomized full-coverage trials against the golden model."""
    start_clock(dut)
    await hw_reset(dut)

    rng = random.Random(0x1247)
    trials = 3 if GL_TEST else 12

    for t in range(trials):
        A = [[rng.randint(-8, 7) for _ in range(K)] for _ in range(N)]
        wbytes = [[rng.randint(0, 255) for _ in range(K // 2)]
                  for _ in range(N)]
        results = await run_matmul(dut, A, wbytes)
        check(dut, results, golden_matmul(A, wbytes), f"trial {t}")
        dut._log.info(f"trial {t} OK")

    dut._log.info(f"random test PASSED ({trials} trials)")


@cocotb.test()
async def test_dense_mode(dut):
    """Dense mode: all select bits = 0 (Roune: 'set the upper 8th bit to zero').
    Only even positions (k=2j) get values; odd positions are always zero.
    This gives dense operation at half throughput — same as NVIDIA's dense
    mode for 2:4 sparsity.
    """
    start_clock(dut)
    await hw_reset(dut)

    A = [[1, 2, 3, 4, 5, 6],
         [-1, -2, -3, -4, -5, -6],
         [7, -8, 7, -8, 7, -8]]

    # All select=0: only k=0,2,4 get values; k=1,3,5 are always zero
    wbytes = [[0x03, 0x05, 0x07],   # col 0: W[0]=3, W[2]=5, W[4]=7
              [0x7E, 0x7C, 0x7A],   # col 1: W[0]=-2, W[2]=-4, W[4]=-6
              [0x01, 0x02, 0x03]]   # col 2: W[0]=1, W[2]=2, W[4]=3

    results = await run_matmul(dut, A, wbytes)
    dut._log.info(f"dense results: {results}")
    check(dut, results, golden_matmul(A, wbytes), "dense")
    dut._log.info("dense mode test PASSED")


@cocotb.test()
async def test_dense_int8_mode(dut):
    """Native int8 dense mode (RUN with dense=1): each weight byte is a
    full int8 for one contraction step (K=3, half throughput), so
    off-the-shelf int8-quantized models map directly. Odd activation
    slots must be ignored — they hold garbage here to prove it.
    """
    start_clock(dut)
    await hw_reset(dut)

    # Real activations at even elements 0,2,4; garbage at odd elements
    A = [[3, -8, -5, 7, 2, -8],
         [-7, 5, 6, -8, -1, 7],
         [4, -3, -8, 6, 7, -2]]

    # Full-range int8 weights, including -128 and 127
    wbytes = [[0x80, 0x7F, 0xFF],   # col 0: -128, 127, -1
              [0x05, 0xFB, 0x40],   # col 1: 5, -5, 64
              [0xC0, 0x2A, 0x93]]   # col 2: -64, 42, -109

    results = await run_matmul(dut, A, wbytes, dense=1)
    dut._log.info(f"int8 dense results: {results}")
    check(dut, results, golden_dense_int8(A, wbytes), "int8-dense")

    # Same operands in sparse mode must interpret bytes as Int7+1
    # (mode is latched per RUN, not sticky)
    results_sparse = await run_matmul(dut, A, wbytes, dense=0)
    check(dut, results_sparse, golden_matmul(A, wbytes), "back-to-sparse")
    dut._log.info("int8 dense mode test PASSED")
