/*
 * Weight memory: 3 columns x 3 Int7+1 bytes.
 *
 * Each byte {select, value[6:0]} covers TWO dense weight positions
 * (contraction steps k=2j and k=2j+1 of column c): value sits at
 * k = 2j + select, the other position is zero. 9 bytes encode a dense
 * 6x3 weight matrix — the 2x storage saving of the Int7+1 format.
 *
 * Read out per column, one byte per wavefront step. Columns read as 0
 * when not enabled (byte 0 decodes to value 0 -> product 0).
 */

`default_nettype none

module memory_b (
    input  wire        clk,
    input  wire        write_enable,
    input  wire [1:0]  write_line,      // column 0..2
    input  wire [1:0]  write_elem,      // pair slot 0..2
    input  wire [7:0]  data_in,
    input  wire [2:0]  read_enable,     // per-column
    input  wire [5:0]  read_slot,       // 2-bit slot index per column (0..2)
    output wire [23:0] data_out         // 3 cols x Int7+1 byte
);

    reg [7:0] mem [0:2][0:2];

    always @(posedge clk) begin
        if (write_enable && write_line < 2'd3 && write_elem < 2'd3)
            mem[write_line][write_elem] <= data_in;
    end

    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : read_col
            wire [1:0] slot = read_slot[2*i +: 2];
            assign data_out[8*i +: 8] = read_enable[i]
                ? mem[i][slot]
                : 8'd0;
        end
    endgenerate

endmodule
