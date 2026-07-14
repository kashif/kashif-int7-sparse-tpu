/*
 * Processing Element: Int7+1 sparse MAC (output-stationary)
 *
 * Dataflow follows the reference mini-TPU PE (activations flow right,
 * weights flow down), but the weight stream carries Int7+1 bytes
 * {select, value[6:0]} and the activation pipe carries an INT4 *pair*
 * (two consecutive contraction steps). The select bit muxes which
 * activation of the pair gets multiplied — a 4-bit mux does the work
 * of a second multiplier, so each cycle advances TWO steps of the
 * contraction (1:2 structured sparsity along k, "sparsity for free").
 *
 * All flops have async reset (dfrtp) — avoids the gate-level
 * X-poisoning documented in the reference REPORT.md for no-reset
 * dfxtp pipeline registers. Affordable on a 1x2 tile.
 */

`default_nettype none

module pe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        we,      // shift pipes + accumulate
    input  wire        clr,     // clear accumulator (start of RUN)
    input  wire [7:0]  a_in,    // activation pair {a_odd[3:0], a_even[3:0]}
    input  wire [7:0]  b_in,    // Int7+1 weight byte {select, value[6:0]}
    output wire [7:0]  a_out,   // pair passed right
    output wire [7:0]  b_out,   // byte passed down
    output wire [11:0] c_out    // accumulated result
);

    reg [7:0] a_reg, b_reg;
    reg signed [11:0] c_reg;

    // Int7+1 decode: select picks the even (k=2j) or odd (k=2j+1)
    // activation of the pair; value is a 7-bit signed integer.
    wire signed [6:0] w_val = $signed(b_in[6:0]);
    wire signed [3:0] act   = b_in[7] ? $signed(a_in[7:4])
                                      : $signed(a_in[3:0]);

    // |value| <= 64, |act| <= 8 -> |product| <= 512, fits signed 11 bits.
    wire signed [10:0] prod = w_val * act;

    // K = 6 contraction: 3 MAC steps, |acc| <= 1536 -> exact in 12 bits.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg <= 8'd0;
            b_reg <= 8'd0;
            c_reg <= 12'sd0;
        end else begin
            if (clr)
                c_reg <= 12'sd0;
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
