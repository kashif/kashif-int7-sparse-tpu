/*
 * SPI slave — ported from the reference mini-TPU spi.v, widened to
 * 16-bit instructions and a 108-bit accumulator readback stream.
 *
 * MOSI shifts in on posedge SCLK while CS is low, LSB-first (bit 0 of
 * the instruction is sent first). When the 16th bit lands, the
 * bit counter wraps 15 -> 0; the clk-domain detector turns that wrap
 * into a single-cycle data_ready pulse that presents the instruction
 * to the control unit for exactly one clk cycle (0 = NOP otherwise).
 *
 * Constraint inherited from the reference: SCLK must be much slower
 * than clk (the bit counter crosses into the clk domain unsynchronised;
 * the reference silicon drives SCLK <= clk/6).
 */

`default_nettype none

module spi (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         mosi,
    input  wire         cs,
    input  wire         sclk,
    input  wire         ready_to_send,
    input  wire [107:0] data_in,

    output reg          miso,
    output wire [15:0]  data_buffer_output
);

    reg [15:0] data_buffer;
    reg [3:0]  bit_counter;
    reg [3:0]  bit_counter_prev;
    reg [6:0]  output_data_bit_counter;
    reg        data_ready;
    reg        is_sending;

    always @(posedge sclk or negedge rst_n) begin
        if (!rst_n) begin
            data_buffer             <= 16'd0;
            bit_counter             <= 4'd0;
            output_data_bit_counter <= 7'd0;
            miso                    <= 1'b0;
        end else begin
            if (bit_counter == 4'd15)
                bit_counter <= 4'd0;
            if (!cs) begin
                if (is_sending) begin
                    miso <= data_in[output_data_bit_counter];
                    output_data_bit_counter <= output_data_bit_counter + 7'd1;
                end
                data_buffer <= {mosi, data_buffer[15:1]};
                if (bit_counter < 4'd15)
                    bit_counter <= bit_counter + 4'd1;
            end else begin
                bit_counter <= 4'd0;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            is_sending       <= 1'b0;
            data_ready       <= 1'b0;
            bit_counter_prev <= 4'd0;
        end else begin
            bit_counter_prev <= bit_counter;
            if (ready_to_send && !cs)
                is_sending <= 1'b1;
            if (is_sending && output_data_bit_counter == 7'd108)
                is_sending <= 1'b0;
            // Pulse data_ready for one clk cycle when the bit counter
            // wraps 15 -> 0 (last instruction bit shifted in).
            if (bit_counter == 4'd0 && bit_counter_prev == 4'd15)
                data_ready <= 1'b1;
            else
                data_ready <= 1'b0;
        end
    end

    assign data_buffer_output = data_ready ? data_buffer : 16'd0;

endmodule
