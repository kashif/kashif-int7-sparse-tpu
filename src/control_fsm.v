/*
 * Control FSM for Int7+1 Sparse TPU
 *
 * Protocol over TT pins (8-bit streaming):
 *   uio_in[7:6] = mode: 00=idle, 01=load, 10=compute, 11=output
 *
 *   LOAD (16 cycles): 1 weight per cycle
 *     ui_in[7:0]  = weight byte {select, value[6:0]}
 *     uio_in[5:2] = load_idx (0-15, row=idx[3:2], col=idx[1:0])
 *
 *   COMPUTE (16 cycles): stream INT4 activations, 4 per cycle (2 per ui_in half)
 *     ui_in[7:4]  = act_col0 (INT4 signed) and act_col1 (INT4 signed)
 *     Actually: 2 activations per cycle, rotate columns each cycle
 *     Cycle 0: ui_in[7:4]=act_col0, ui_in[3:0]=act_col1
 *     Cycle 1: ui_in[7:4]=act_col2, ui_in[3:0]=act_col3
 *     Repeat for 16 activation slots = 8 pairs of cycles = 32 total
 *     
 *     Simpler approach: all 4 columns get same activation (broadcast)
 *     ui_in[7:4] = act_a (INT4), ui_in[3:0] = act_b (INT4)
 *     16 cycles, each cycle feeds act to columns in rotation
 *
 *     Simplest: 4 INT4 activations per cycle packed into ui_in
 *     ui_in[7:6] = act_col0 (2-bit, but INT4 needs 4 bits...)
 *     
 *     INT4 = 4 bits. 4 columns × 4 bits = 16 bits. ui_in = 8 bits.
 *     So: 2 columns per cycle, 2 cycles to feed all 4.
 *     ui_in[7:4] = act_col0, ui_in[3:0] = act_col1 (cycle A)
 *     ui_in[7:4] = act_col2, ui_in[3:0] = act_col3 (cycle B)
 *     16 activations per column × 2 cycles = 32 compute cycles
 *
 *   OUTPUT (48 cycles): 16 results × 3 bytes each (12-bit acc)
 *     uo_out = result bytes (high, mid, low per accumulator)
 *     uio_out[7] = done
 *
 * References:
 *   - PFW TPU control: github.com/wangantian/pfw_tpu/src/control_unit.v
 *   - Mini-TPU control: github.com/MILOUDIAS/IEEE_ttsky_mini_tpu_spi
 *   - TT HDL guide: no initial blocks, explicit rst_n, all outputs assigned
 */

`default_nettype none

module control_fsm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  mode,

    // Load interface
    input  wire [7:0]  load_weight,   // ui_in[7:0]
    input  wire [3:0]  load_idx,      // uio_in[5:2]

    // Compute interface
    input  wire signed [3:0] act_a,   // ui_in[7:4] sign-extended
    input  wire signed [3:0] act_b,   // ui_in[3:0] sign-extended
    input  wire        relu_en,       // uio_in[0]

    // Output
    output reg  [7:0]  result_byte,
    output reg         done,

    // Array control
    output wire        weight_load,
    output wire [7:0]  weight_out,
    output wire [3:0]  load_idx_out,
    output wire        act_valid,
    output wire signed [3:0] act_col0, act_col1, act_col2, act_col3,
    output wire        acc_clear,

    // Accumulator readback
    input  wire [11:0] acc00, acc01, acc02, acc03,
    input  wire [11:0] acc10, acc11, acc12, acc13,
    input  wire [11:0] acc20, acc21, acc22, acc23,
    input  wire [11:0] acc30, acc31, acc32, acc33,

    output reg  [3:0]  status
);

    localparam [1:0] MODE_IDLE    = 2'b00;
    localparam [1:0] MODE_LOAD    = 2'b01;
    localparam [1:0] MODE_COMPUTE = 2'b10;
    localparam [1:0] MODE_OUTPUT  = 2'b11;

    // Compute: 16 activation cycles + 1 clear + 1 drain + 1 snapshot = 19
    localparam [5:0] CNT_CLEAR     = 6'd0;
    localparam [5:0] CNT_COMPUTE_LAST = 6'd17;  // 17 compute (same as NVFP4)
    localparam [5:0] CNT_DRAIN     = 6'd18;
    localparam [5:0] CNT_SNAPSHOT  = 6'd19;

    reg [5:0] compute_cnt;
    reg [5:0] output_cnt;
    reg       relu_en_latched;
    reg       sub_cycle;  // 0 = feed cols 0,1; 1 = feed cols 2,3

    // Latched accumulators
    reg [11:0] acc_snap [0:15];

    // ReLU applied
    wire [11:0] acc_relu [0:15];
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : RELU
            assign acc_relu[gi] = (relu_en_latched && acc_snap[gi][11]) ? 12'd0 : acc_snap[gi];
        end
    endgenerate

    // 16:1 output mux
    reg [11:0] acc_read;
    always @(*) begin
        case (output_cnt[5:2])  // 4 bits for 16 entries, 3 bytes each
            4'd0:  acc_read = acc_relu[0];
            4'd1:  acc_read = acc_relu[1];
            4'd2:  acc_read = acc_relu[2];
            4'd3:  acc_read = acc_relu[3];
            4'd4:  acc_read = acc_relu[4];
            4'd5:  acc_read = acc_relu[5];
            4'd6:  acc_read = acc_relu[6];
            4'd7:  acc_read = acc_relu[7];
            4'd8:  acc_read = acc_relu[8];
            4'd9:  acc_read = acc_relu[9];
            4'd10: acc_read = acc_relu[10];
            4'd11: acc_read = acc_relu[11];
            4'd12: acc_read = acc_relu[12];
            4'd13: acc_read = acc_relu[13];
            4'd14: acc_read = acc_relu[14];
            default: acc_read = acc_relu[15];
        endcase
    end

    // Combinational pass-through for load (must be same-cycle)
    assign weight_load  = (mode == MODE_LOAD);
    assign weight_out   = load_weight;
    assign load_idx_out = load_idx;

    // Registered act_valid, acc_clear, and activation routing
    reg act_valid_r;
    reg acc_clear_r;
    assign act_valid = act_valid_r;
    assign acc_clear = acc_clear_r;

    // Registered activations (aligned with act_valid_r)
    reg signed [3:0] act_a_r, act_b_r;
    assign act_col0 = act_a_r;
    assign act_col1 = act_b_r;
    assign act_col2 = act_a_r;
    assign act_col3 = act_b_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_cnt     <= CNT_CLEAR;
            output_cnt      <= 6'd0;
            act_valid_r     <= 1'b0;
            acc_clear_r     <= 1'b0;
            act_a_r         <= 4'sd0;
            act_b_r         <= 4'sd0;
            result_byte     <= 8'd0;
            done            <= 1'b0;
            status          <= 4'd0;
            relu_en_latched <= 1'b0;
        end else begin
            act_valid_r <= 1'b0;
            acc_clear_r <= 1'b0;
            done        <= 1'b0;

            case (mode)
                MODE_LOAD: begin
                    status <= {2'b01, load_idx[3:0]};
                end

                MODE_COMPUTE: begin
                    if (compute_cnt == CNT_CLEAR) begin
                        acc_clear_r     <= 1'b1;
                        act_valid_r     <= 1'b1;
                        act_a_r         <= act_a;
                        act_b_r         <= act_b;
                        compute_cnt     <= 6'd1;
                        relu_en_latched <= relu_en;
                        status          <= 4'b1000;
                    end else if (compute_cnt <= CNT_COMPUTE_LAST) begin
                        act_valid_r <= 1'b1;
                        act_a_r     <= act_a;
                        act_b_r     <= act_b;
                        compute_cnt <= compute_cnt + 6'd1;
                        status      <= {2'b10, compute_cnt[3:0]};
                    end else if (compute_cnt == CNT_DRAIN) begin
                        compute_cnt <= CNT_SNAPSHOT;
                        status      <= 4'b1001;
                    end else begin
                        compute_cnt <= CNT_CLEAR;
                        status      <= 4'b1010;
                        acc_snap[0]  <= acc00;  acc_snap[1]  <= acc01;
                        acc_snap[2]  <= acc02;  acc_snap[3]  <= acc03;
                        acc_snap[4]  <= acc10;  acc_snap[5]  <= acc11;
                        acc_snap[6]  <= acc12;  acc_snap[7]  <= acc13;
                        acc_snap[8]  <= acc20;  acc_snap[9]  <= acc21;
                        acc_snap[10] <= acc22;  acc_snap[11] <= acc23;
                        acc_snap[12] <= acc30;  acc_snap[13] <= acc31;
                        acc_snap[14] <= acc32;  acc_snap[15] <= acc33;
                    end
                end

                MODE_OUTPUT: begin
                    // 16 results × 3 bytes = 48 cycles
                    if (output_cnt < 6'd48) begin
                        // 12-bit acc: byte 0 = {4'b0, acc[11:8]}, byte 1 = acc[7:0]
                        // Actually 12 bits = 2 bytes: {4'b0, acc[11:8]} and acc[7:0]
                        // 16 × 2 = 32 cycles (simpler than 3-byte packing)
                        if (output_cnt < 6'd32) begin
                            result_byte <= output_cnt[0] ? acc_read[7:0]
                                                       : {4'b0, acc_read[11:8]};
                        end
                        output_cnt <= output_cnt + 6'd1;
                        status     <= {2'b11, output_cnt[3:0]};
                    end else begin
                        done       <= 1'b1;
                        output_cnt <= 6'd0;
                        status     <= 4'b1100;
                    end
                end

                default: begin
                    status <= 4'b0000;
                end
            endcase
        end
    end

endmodule
