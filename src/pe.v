/*
 * Processing Element: Int7+1 sparse MAC (output-stationary)
 * with native int8 dense mode.
 *
 * Dataflow follows the reference mini-TPU PE (activations flow right,
 * weights flow down). Two weight formats, selected by `dense`:
 *
 *  Sparse (dense=0): b_in = {select, value[6:0]} Int7+1. The activation
 *    pipe carries an INT4 pair (two consecutive contraction steps) and
 *    the select bit muxes which one gets multiplied — a 4-bit mux does
 *    the work of a second multiplier, so each cycle advances TWO steps
 *    of the contraction (1:2 structured sparsity, "sparsity for free").
 *
 *  Dense (dense=1): b_in is a full int8 weight for ONE contraction
 *    step; only the even activation of the pair is used. Half the
 *    throughput of sparse mode on the same hardware — mirroring how
 *    NVIDIA runs dense at half rate on 2:4-sparse tensor cores, and
 *    giving off-the-shelf int8-quantized models a native path.
 *
 * All flops have async reset (dfrtp) — avoids the gate-level
 * X-poisoning documented in the reference REPORT.md for no-reset
 * dfxtp pipeline registers.
 */

`default_nettype none

module pe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        we,      // shift pipes + accumulate
    input  wire        clr,     // clear accumulator (start of RUN)
    input  wire        dense,   // 0 = Int7+1 sparse, 1 = int8 dense
    input  wire [7:0]  a_in,    // activation pair {a_odd[3:0], a_even[3:0]}
    input  wire [7:0]  b_in,    // weight byte (Int7+1 or int8)
    output wire [7:0]  a_out,   // pair passed right
    output wire [7:0]  b_out,   // byte passed down
    output wire [12:0] c_out    // accumulated result
);

    reg [7:0] a_reg, b_reg;
    reg signed [12:0] c_reg;

    // Weight: full int8 in dense mode, sign-extended Int7 value in sparse.
    wire signed [7:0] w8 = dense ? $signed(b_in)
                                 : $signed({b_in[6], b_in[6:0]});

    // Activation: dense always uses the even slot; sparse muxes by select.
    wire signed [3:0] act = (dense || !b_in[7]) ? $signed(a_in[3:0])
                                                : $signed(a_in[7:4]);

    // |w| <= 128, |act| <= 8 -> |product| <= 1024, fits signed 12 bits.
    wire signed [11:0] prod = w8 * act;

    // 3 MAC steps: sparse max |acc| = 1536, dense max = 3072 —
    // both exact in a 13-bit signed accumulator.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg <= 8'd0;
            b_reg <= 8'd0;
            c_reg <= 13'sd0;
        end else begin
            if (clr)
                c_reg <= 13'sd0;
            else if (we)
                c_reg <= c_reg + prod;
            if (we) begin
                a_reg <= a_in;
                b_reg <= b_in;
            end
        end
    end

    assign a_out = a_reg;
    assign b_out = b_reg;
    assign c_out = c_reg;

endmodule
