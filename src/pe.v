/*
 * Processing Element: Int7+1 Sparse MAC
 *
 * Weight encoding (8 bits): {select, value[6:0]}
 * The select bit is consumed during weight loading — only the PE
 * whose column parity matches the select bit gets the weight value.
 * The other PE in the pair gets 0 (sparsity baked in at load time).
 *
 * This eliminates the is_odd port — sparsity is resolved at load time,
 * not compute time. Simpler hardware, no runtime gating needed.
 *
 * References:
 *   - Roune: "7-bit integer multiplier with 1:2 structured sparsity"
 *   - NVIDIA 2:4 sparsity: arXiv:2104.08378
 *   - TT HDL guide: minimal flops, no initial blocks, explicit rst_n
 */

`default_nettype none

module pe #(
    parameter IS_ODD = 1'b0
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        weight_load,
    input  wire [7:0]  weight_in,
    input  wire        act_valid,
    input  wire signed [3:0] act_in,
    input  wire        acc_clear,
    output wire [11:0] acc_out
);

    // Weight register: store value only if select matches is_odd
    // If select doesn't match, store 0 (this PE is the sparse zero)
    reg signed [6:0] weight_val;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            weight_val <= 7'sd0;
        else if (weight_load) begin
            if (weight_in[7] == IS_ODD)
                weight_val <= $signed(weight_in[6:0]);
            else
                weight_val <= 7'sd0;
        end
    end

    // 7x4 signed multiply
    wire signed [10:0] product;
    assign product = weight_val * $signed(act_in);

    // 12-bit signed accumulator
    reg signed [11:0] acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 12'sd0;
        else if (acc_clear)
            acc <= act_valid ? $signed(product) : 12'sd0;
        else if (act_valid)
            acc <= acc + product;
    end

    assign acc_out = acc;

endmodule
