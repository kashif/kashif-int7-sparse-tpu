/*
 * TT Int7+1 Sparse Mini-TPU — Tiny Tapeout top level (1x2 tile)
 *
 * 3x3 output-stationary systolic array with Int7+1 weights: each
 * 8-bit weight byte {select, value[6:0]} covers two consecutive
 * contraction steps (1:2 structured sparsity along k). The select
 * bit muxes which INT4 activation of a pair gets multiplied — a
 * 4-bit mux replaces a second multiplier, so every cycle advances
 * two contraction steps. 12 weight bytes encode a dense 8x3 matrix
 * (K=8 — tiles power-of-two model dimensions with no padding).
 *
 * Architecture and SPI protocol follow the proven reference design
 * (github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi), widened to 16-bit
 * instructions. See docs/info.md for the ISA.
 *
 * Pinout:
 *   ui_in[0] = MOSI, ui_in[1] = CS (active low), ui_in[2] = SCLK
 *   uo_out   = result byte (selected by STORE)
 *   uio[0]   = MISO (output), uio[1] = ready (output), rest unused
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

    wire         mosi = ui_in[0];
    wire         cs   = ui_in[1];
    wire         sclk = ui_in[2];

    wire [15:0]  instruction;
    wire         ready_to_send;
    wire [125:0] array_data_out;
    wire         miso;

    tpu u_tpu (
        .clk            (clk),
        .rst_n          (rst_n),
        .instruction    (instruction),
        .ready_to_send  (ready_to_send),
        .result         (uo_out),
        .array_data_out (array_data_out)
    );

    spi u_spi (
        .clk                (clk),
        .rst_n              (rst_n),
        .mosi               (mosi),
        .cs                 (cs),
        .sclk               (sclk),
        .ready_to_send      (ready_to_send),
        .data_in            (array_data_out),
        .miso               (miso),
        .data_buffer_output (instruction)
    );

    // Drive uio_oe/uio_out through (* keep *) flip-flops instead of
    // constants: direct constant assigns synthesize to conb cells whose
    // pulldown nets magic merges with VGND during extraction, producing
    // LVS mismatches (documented in the reference REPORT.md §conb).
    // uio[0] = MISO out, uio[1] = ready out, uio[7:2] = inputs (oe 0).
    (* keep = "true" *) reg [7:0] uio_oe_q;
    (* keep = "true" *) reg [5:0] uio_out_high_q;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uio_oe_q       <= 8'b0;
            uio_out_high_q <= 6'b0;
        end else begin
            uio_oe_q       <= 8'b0000_0011;
            uio_out_high_q <= 6'b0;
        end
    end
    assign uio_oe  = uio_oe_q;
    assign uio_out = {uio_out_high_q, ready_to_send, miso};

    wire _unused = &{ena, ui_in[7:3], uio_in, 1'b0};

endmodule
