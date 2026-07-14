/*
 * 4x4 Weight-Stationary Systolic Array for Int7+1 Sparse TPU
 *
 * 16 PEs in 4x4 grid. Columns paired: (0,1) and (2,3).
 * Each 8-bit weight has {select, value[6:0]}. Select bit picks
 * which of the pair is non-zero — 50% structured sparsity.
 *
 * References:
 *   - NVIDIA 2:4 sparsity: arXiv:2104.08378
 *   - PFW TPU: github.com/wangantian/pfw_tpu
 *   - Mini-TPU: github.com/MILOODIAS/IEEE_ttsky_mini_tpu_spi
 */

`default_nettype none

module systolic_array_4x4 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        weight_load,
    input  wire [7:0]  weight_in,
    input  wire [3:0]  load_idx,
    input  wire        act_valid,
    input  wire signed [3:0] act_col0,
    input  wire signed [3:0] act_col1,
    input  wire signed [3:0] act_col2,
    input  wire signed [3:0] act_col3,
    input  wire        acc_clear,
    output wire [11:0] acc00, acc01, acc02, acc03,
    output wire [11:0] acc10, acc11, acc12, acc13,
    output wire [11:0] acc20, acc21, acc22, acc23,
    output wire [11:0] acc30, acc31, acc32, acc33
);

    wire [1:0] load_row = load_idx[3:2];
    wire [1:0] load_col = load_idx[1:0];

    wire [11:0] pe_acc [0:3][0:3];

    // Odd/even PEs via parameter
    pe #(.IS_ODD(1'b0)) pe00 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd0 && load_col==2'd0), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col0), .acc_clear(acc_clear), .acc_out(pe_acc[0][0]));
    pe #(.IS_ODD(1'b1)) pe01 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd0 && load_col==2'd1), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col1), .acc_clear(acc_clear), .acc_out(pe_acc[0][1]));
    pe #(.IS_ODD(1'b0)) pe02 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd0 && load_col==2'd2), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col2), .acc_clear(acc_clear), .acc_out(pe_acc[0][2]));
    pe #(.IS_ODD(1'b1)) pe03 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd0 && load_col==2'd3), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col3), .acc_clear(acc_clear), .acc_out(pe_acc[0][3]));
    pe #(.IS_ODD(1'b0)) pe10 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd1 && load_col==2'd0), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col0), .acc_clear(acc_clear), .acc_out(pe_acc[1][0]));
    pe #(.IS_ODD(1'b1)) pe11 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd1 && load_col==2'd1), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col1), .acc_clear(acc_clear), .acc_out(pe_acc[1][1]));
    pe #(.IS_ODD(1'b0)) pe12 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd1 && load_col==2'd2), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col2), .acc_clear(acc_clear), .acc_out(pe_acc[1][2]));
    pe #(.IS_ODD(1'b1)) pe13 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd1 && load_col==2'd3), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col3), .acc_clear(acc_clear), .acc_out(pe_acc[1][3]));
    pe #(.IS_ODD(1'b0)) pe20 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd2 && load_col==2'd0), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col0), .acc_clear(acc_clear), .acc_out(pe_acc[2][0]));
    pe #(.IS_ODD(1'b1)) pe21 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd2 && load_col==2'd1), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col1), .acc_clear(acc_clear), .acc_out(pe_acc[2][1]));
    pe #(.IS_ODD(1'b0)) pe22 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd2 && load_col==2'd2), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col2), .acc_clear(acc_clear), .acc_out(pe_acc[2][2]));
    pe #(.IS_ODD(1'b1)) pe23 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd2 && load_col==2'd3), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col3), .acc_clear(acc_clear), .acc_out(pe_acc[2][3]));
    pe #(.IS_ODD(1'b0)) pe30 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd3 && load_col==2'd0), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col0), .acc_clear(acc_clear), .acc_out(pe_acc[3][0]));
    pe #(.IS_ODD(1'b1)) pe31 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd3 && load_col==2'd1), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col1), .acc_clear(acc_clear), .acc_out(pe_acc[3][1]));
    pe #(.IS_ODD(1'b0)) pe32 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd3 && load_col==2'd2), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col2), .acc_clear(acc_clear), .acc_out(pe_acc[3][2]));
    pe #(.IS_ODD(1'b1)) pe33 (.clk(clk), .rst_n(rst_n), .weight_load(weight_load && load_row==2'd3 && load_col==2'd3), .weight_in(weight_in), .act_valid(act_valid), .act_in(act_col3), .acc_clear(acc_clear), .acc_out(pe_acc[3][3]));

    assign acc00 = pe_acc[0][0];  assign acc01 = pe_acc[0][1];
    assign acc02 = pe_acc[0][2];  assign acc03 = pe_acc[0][3];
    assign acc10 = pe_acc[1][0];  assign acc11 = pe_acc[1][1];
    assign acc12 = pe_acc[1][2];  assign acc13 = pe_acc[1][3];
    assign acc20 = pe_acc[2][0];  assign acc21 = pe_acc[2][1];
    assign acc22 = pe_acc[2][2];  assign acc23 = pe_acc[2][3];
    assign acc30 = pe_acc[3][0];  assign acc31 = pe_acc[3][1];
    assign acc32 = pe_acc[3][2];  assign acc33 = pe_acc[3][3];

endmodule
