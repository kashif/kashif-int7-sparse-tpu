/*
 * 3x3 output-stationary systolic array, Int7+1 sparse edition.
 *
 * Structure follows the reference mini-TPU array.v: activations flow
 * right, weights flow down, results accumulate in place. Differences:
 *   - horizontal pipes carry an INT4 *pair* (8 bits) — two contraction
 *     steps per cycle
 *   - vertical pipes carry Int7+1 weight bytes (8 bits)
 *   - 14-bit accumulators (exact for K=8, no truncation)
 *   - clr input zeroes all accumulators at the start of a RUN
 */

`default_nettype none

module array (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         we,
    input  wire         clr,
    input  wire         dense,    // 0 = Int7+1 sparse, 1 = int8 dense

    input  wire [23:0]  a_in,     // 3 rows x activation pair
    input  wire [23:0]  b_in,     // 3 cols x weight byte
    output wire [125:0] data_out  // 9 accumulators x 14 bits, row-major
);

    // a_pipe[row][col]: pair flowing right; b_pipe[row][col]: byte flowing down
    wire [7:0]  a_pipe [0:2][0:3];
    wire [7:0]  b_pipe [0:3][0:2];
    wire [13:0] c_bus  [0:2][0:2];

    genvar row, col;
    generate
        for (row = 0; row < 3; row = row + 1) begin : map_a_in
            assign a_pipe[row][0] = a_in[8*row +: 8];
        end
        for (col = 0; col < 3; col = col + 1) begin : map_b_in
            assign b_pipe[0][col] = b_in[8*col +: 8];
        end
    endgenerate

    generate
        for (row = 0; row < 3; row = row + 1) begin : ROWS
            for (col = 0; col < 3; col = col + 1) begin : COLS
                pe pe_inst (
                    .clk   (clk),
                    .rst_n (rst_n),
                    .we    (we),
                    .clr   (clr),
                    .dense (dense),
                    .a_in  (a_pipe[row][col]),
                    .b_in  (b_pipe[row][col]),
                    .a_out (a_pipe[row][col+1]),
                    .b_out (b_pipe[row+1][col]),
                    .c_out (c_bus [row][col])
                );
            end
        end
    endgenerate

    generate
        for (row = 0; row < 3; row = row + 1) begin : flat_row
            for (col = 0; col < 3; col = col + 1) begin : flat_col
                assign data_out[14*(row*3+col) +: 14] = c_bus[row][col];
            end
        end
    endgenerate

endmodule
