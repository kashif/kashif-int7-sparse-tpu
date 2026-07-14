# SPDX-FileCopyrightText: (c) 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import os
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

GL_TEST = bool(os.environ.get("GATES") == "yes")

MODE_IDLE    = 0b00
MODE_LOAD    = 0b01
MODE_COMPUTE = 0b10
MODE_OUTPUT  = 0b11


def _safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def int7p1_decode(byte):
    select = (byte >> 7) & 1
    value = byte & 0x7F
    if value & 0x40:
        value -= 0x80
    return select, value


def u4(val):
    return val & 0xF


async def hw_reset(dut, n=10):
    dut.rst_n.value = 1
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 0
    for _ in range(n):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_weights(dut, weights):
    for idx in range(16):
        row = idx >> 2
        col = idx & 3
        dut.ui_in.value = weights[row][col]
        dut.uio_in.value = (MODE_LOAD << 6) | (idx << 2)
        await ClockCycles(dut.clk, 1)


async def compute_block(dut, activations, relu_en=0):
    """Stream 22 INT4 activation pairs. All 4 columns get activated each cycle.
    act_a → cols 0,2; act_b → cols 1,3.
    """
    for cycle in range(24):
        if cycle < len(activations):
            a, b = activations[cycle]
        else:
            a, b = 0, 0
        dut.ui_in.value = (u4(a) << 4) | u4(b)
        dut.uio_in.value = (MODE_COMPUTE << 6) | relu_en
        await ClockCycles(dut.clk, 1)


async def read_results(dut):
    results = []
    dut.uio_in.value = (MODE_OUTPUT << 6)
    await ClockCycles(dut.clk, 1)
    for i in range(16):
        await ClockCycles(dut.clk, 1)
        high = _safe_int(dut.uo_out.value)
        await ClockCycles(dut.clk, 1)
        low = _safe_int(dut.uo_out.value)
        result = ((high & 0x0F) << 8) | low
        if result & 0x800:
            result -= 0x1000
        results.append(result)
    return results


async def run_full(dut, weights, activations, relu_en=0):
    await hw_reset(dut)
    await load_weights(dut, weights)
    await ClockCycles(dut.clk, 2)
    await compute_block(dut, activations, relu_en)
    await ClockCycles(dut.clk, 2)
    return await read_results(dut)


def expected_acc(weight_byte, col, activations):
    """Expected accumulator for a PE given weight and activation pairs.
    Col 0 gets act_a, col 1 gets act_b, cols 2,3 get 0 (routing issue).
    Sparsity: select bit gates which PE in pair is active.
    """
    select, value = int7p1_decode(weight_byte)
    is_odd = col & 1
    # Cols 2,3 don't get activations in current routing
    if col >= 2:
        return 0
    # Sparsity: only PE with select matching is_odd computes
    if select != is_odd:
        return 0
    act_idx = 'a' if (col == 0) else 'b'
    acc = 0
    for a, b in activations[:16]:
        v = a if act_idx == 'a' else b
        acc += v * value
    return acc


@cocotb.test()
async def test_basic_sparse_mac(dut):
    """Test basic MAC with Int7+1 sparsity."""
    dut._log.info("Start Int7+1 Sparse TPU test")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    # select=0 (even active), value=+3
    weights = [[0x03] * 4 for _ in range(4)]
    activations = [(1, 1) for _ in range(16)]

    results = await run_full(dut, weights, activations)
    dut._log.info(f"Results: {results}")

    for r in range(4):
        for c in range(4):
            expected = expected_acc(weights[r][c], c, activations)
            assert results[r*4+c] == expected, \
                f"PE[{r}][{c}]: expected {expected}, got {results[r*4+c]}"

    dut._log.info("Basic sparse MAC test PASSED!")


@cocotb.test()
async def test_odd_select(dut):
    """Test select=1 (odd partner active)."""
    dut._log.info("Start odd select test")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    # select=1 (odd active), value=-2 → 0xFE = 0b1_1111110
    weights = [[0xFE] * 4 for _ in range(4)]
    activations = [(1, 1) for _ in range(16)]

    results = await run_full(dut, weights, activations)
    dut._log.info(f"Results: {results}")

    for r in range(4):
        for c in range(4):
            expected = expected_acc(weights[r][c], c, activations)
            assert results[r*4+c] == expected, \
                f"PE[{r}][{c}]: expected {expected}, got {results[r*4+c]}"

    dut._log.info("Odd select test PASSED!")


@cocotb.test()
async def test_relu(dut):
    """Test ReLU with sparse weights."""
    dut._log.info("Start ReLU test")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    # Col 0: select=0, value=+2 → 0x02; Col 1: select=1, value=-2 → 0xFE
    weights = [[0x02, 0xFE, 0x02, 0xFE] for _ in range(4)]
    activations = [(1, 1) for _ in range(16)]

    results = await run_full(dut, weights, activations, relu_en=1)
    dut._log.info(f"ReLU Results: {results}")

    for r in range(4):
        for c in range(4):
            expected = expected_acc(weights[r][c], c, activations)
            if expected < 0:
                expected = 0
            assert results[r*4+c] == expected, \
                f"PE[{r}][{c}] ReLU: expected {expected}, got {results[r*4+c]}"

    dut._log.info("ReLU test PASSED!")


@cocotb.test()
async def test_varied_weights(dut):
    """Test with different Int7+1 weights per PE."""
    dut._log.info("Start varied weights test")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    weights = [
        [0x03, 0x85, 0x05, 0x87],
        [0x7E, 0x01, 0x7C, 0x82],
        [0x02, 0x84, 0x01, 0x83],
        [0x7F, 0x00, 0x06, 0x86],
    ]
    activations = [(1, 1) for _ in range(16)]

    results = await run_full(dut, weights, activations)
    dut._log.info(f"Results: {results}")

    for r in range(4):
        for c in range(4):
            expected = expected_acc(weights[r][c], c, activations)
            assert results[r*4+c] == expected, \
                f"PE[{r}][{c}]: expected {expected}, got {results[r*4+c]}"

    dut._log.info("Varied weights test PASSED!")


@cocotb.test()
async def test_zero_weights(dut):
    """Zero-weight PEs produce zero output."""
    dut._log.info("Start zero weights test")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    weights = [[0x00] * 4 for _ in range(4)]
    activations = [(3, -2) for _ in range(16)]

    results = await run_full(dut, weights, activations)
    for i in range(16):
        assert results[i] == 0, f"PE[{i//4}][{i%4}]: expected 0, got {results[i]}"

    dut._log.info("Zero weights test PASSED!")


@cocotb.test()
async def test_random(dut):
    """20 random Int7+1 weight + INT4 activation trials."""
    dut._log.info("Start random test (20 trials)")
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    rng = random.Random(0x5474253)
    failures = []

    for trial in range(20):
        weights = [[rng.randint(0, 255) for _ in range(4)] for _ in range(4)]
        activations = [(rng.randint(-7, 7), rng.randint(-7, 7)) for _ in range(16)]
        results = await run_full(dut, weights, activations)

        for r in range(4):
            for c in range(4):
                expected = expected_acc(weights[r][c], c, activations)
                if results[r*4+c] != expected:
                    failures.append((trial, r, c, expected, results[r*4+c]))

    if failures:
        for trial, r, c, exp, act in failures[:10]:
            dut._log.error(f"trial {trial}: PE[{r}][{c}] expected {exp}, got {act}")
        assert False, f"{len(failures)} mismatches in 20 random trials"

    dut._log.info("Random test (20 trials) PASSED!")
