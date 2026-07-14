/*
 * TT Int7+1 Sparse TPU — Top-Level Tiny Tapeout Module
 *
 * 4x4 weight-stationary systolic array with Int7+1 structured sparsity.
 * Each 8-bit weight encodes {select, value[6:0]} — the select bit
 * picks which of two adjacent weight positions is non-zero. 50%
 * sparsity baked into the data format, no pruning needed.
 *
 * Concept by Roune: "A 7-bit integer multiplier with 1:2 structured
 * sparsity baked in. The 8th bit repurposed to encode which of two
 * adjacent entries is non-zero."
 *
 * Protocol:
 *   uio_in[7:6] = mode: 00=idle, 01=load, 10=compute, 11=output
 *
 *   LOAD (16 cycles): ui_in=weight_byte, uio_in[5:2]=load_idx(0-15)
 *   COMPUTE (34 cycles): ui_in[7:4]=act_a(INT4), ui_in[3:0]=act_b(INT4)
 *     uio_in[0]=relu_en. Alternates cols (0,1) then (2,3) each cycle.
 *   OUTPUT (32 cycles): uo_out=result bytes, uio_out[7]=done
 *
 * HDL guide compliance (tinytapeout.com/hdl/important/):
 *   - Exact module port definition
 *   - No initial blocks; explicit rst_n
 *   - All outputs assigned; (* keep *) FFs for LVS safety
 *   - _unused wire; default_nettype none
 */

`default_nettype none

module tt_um_kashif_int7_sparse_tpu (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    wire [1:0] mode = uio_in[7:6];

    // Load interface
    wire [7:0] load_weight = ui_in;
    wire [3:0] load_idx    = uio_in[5:2];

    // Compute interface: INT4 sign-extend from 4-bit ui_in halves
    wire signed [3:0] act_a = $signed(ui_in[7:4]);
    wire signed [3:0] act_b = $signed(ui_in[3:0]);
    wire       relu_en  = uio_in[0];

    // Control <-> Array
    wire        weight_load;
    wire [7:0]  weight_out;
    wire [3:0]  load_idx_w;
    wire        act_valid;
    wire signed [3:0] act_col0_w, act_col1_w, act_col2_w, act_col3_w;
    wire        acc_clear;

    wire [11:0] acc00, acc01, acc02, acc03;
    wire [11:0] acc10, acc11, acc12, acc13;
    wire [11:0] acc20, acc21, acc22, acc23;
    wire [11:0] acc30, acc31, acc32, acc33;

    wire [7:0] result_byte;
    wire       done;
    wire [3:0] status;

    control_fsm u_ctrl (
        .clk            (clk),
        .rst_n          (rst_n),
        .mode           (mode),
        .load_weight    (load_weight),
        .load_idx       (load_idx),
        .act_a          (act_a),
        .act_b          (act_b),
        .relu_en        (relu_en),
        .result_byte    (result_byte),
        .done           (done),
        .weight_load    (weight_load),
        .weight_out     (weight_out),
        .load_idx_out   (load_idx_w),
        .act_valid      (act_valid),
        .act_col0       (act_col0_w),
        .act_col1       (act_col1_w),
        .act_col2       (act_col2_w),
        .act_col3       (act_col3_w),
        .acc_clear      (acc_clear),
        .acc00          (acc00),  .acc01          (acc01),
        .acc02          (acc02),  .acc03          (acc03),
        .acc10          (acc10),  .acc11          (acc11),
        .acc12          (acc12),  .acc13          (acc13),
        .acc20          (acc20),  .acc21          (acc21),
        .acc22          (acc22),  .acc23          (acc23),
        .acc30          (acc30),  .acc31          (acc31),
        .acc32          (acc32),  .acc33          (acc33),
        .status         (status)
    );

    systolic_array_4x4 u_array (
        .clk         (clk),
        .rst_n       (rst_n),
        .weight_load (weight_load),
        .weight_in   (weight_out),
        .load_idx    (load_idx_w),
        .act_valid   (act_valid),
        .act_col0    (act_col0_w),
        .act_col1    (act_col1_w),
        .act_col2    (act_col2_w),
        .act_col3    (act_col3_w),
        .acc_clear   (acc_clear),
        .acc00       (acc00),  .acc01       (acc01),
        .acc02       (acc02),  .acc03       (acc03),
        .acc10       (acc10),  .acc11       (acc11),
        .acc12       (acc12),  .acc13       (acc13),
        .acc20       (acc20),  .acc21       (acc21),
        .acc22       (acc22),  .acc23       (acc23),
        .acc30       (acc30),  .acc31       (acc31),
        .acc32       (acc32),  .acc33       (acc33)
    );

    assign uo_out = result_byte;

    (* keep = "true" *) reg [7:0] uio_oe_q;
    (* keep = "true" *) reg [7:0] uio_out_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uio_oe_q  <= 8'b0;
            uio_out_q <= 8'b0;
        end else begin
            uio_oe_q  <= (mode == 2'b11) ? 8'b1111_1111 : 8'b0000_0000;
            uio_out_q <= {done, 3'b0, status};
        end
    end
    assign uio_oe  = uio_oe_q;
    assign uio_out = uio_out_q;

    wire _unused = &{ena, 1'b0};

endmodule
