module systolic_array_8x8 #(
   parameter DATA_WIDTH = 16,
   parameter ACC_WIDTH  = 32,
   parameter SIZE       = 8
)(
   input  logic clk,
   input  logic rst,
   input  logic signed [SIZE*DATA_WIDTH-1:0]      a_in,
   input  logic signed [SIZE*DATA_WIDTH-1:0]      b_in,
   output logic signed [SIZE*SIZE*ACC_WIDTH-1:0]  acc_out,
   output logic                                   done
);
 
   localparam DONE_CYCLE = 3*SIZE - 2;
 
   // --- Cycle counter --------------------------------------------------
   logic [$clog2(DONE_CYCLE+1)-1:0] cnt;
 
   always_ff @(posedge clk) begin
       if (rst)
           cnt <= '0;
       else if (cnt < DONE_CYCLE[$clog2(DONE_CYCLE+1)-1:0])
           cnt <= cnt + 1;
   end
 
   assign done = (cnt == DONE_CYCLE[$clog2(DONE_CYCLE+1)-1:0]);
 
   // --- Input gating (zero after SIZE feed cycles) ---------------------
   logic signed [DATA_WIDTH-1:0] a_gated [0:SIZE-1];
   logic signed [DATA_WIDTH-1:0] b_gated [0:SIZE-1];
 
   genvar i;
   generate
       for (i = 0; i < SIZE; i++) begin : gate_inputs
           assign a_gated[i] = (cnt < SIZE[$clog2(DONE_CYCLE+1)-1:0])
                                ? a_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] : '0;
           assign b_gated[i] = (cnt < SIZE[$clog2(DONE_CYCLE+1)-1:0])
                                ? b_in[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] : '0;
       end
   endgenerate
 
   // --- Inter-PE wires -------------------------------------------------
   logic signed [DATA_WIDTH-1:0] a_bus [0:SIZE-1][0:SIZE];
   logic signed [DATA_WIDTH-1:0] b_bus [0:SIZE][0:SIZE-1];
 
   // --- Input skew registers -------------------------------------------
   // Row r of A is delayed by r cycles; Column c of B is delayed by c cycles.
   // This aligns partial products so A[r][k]*B[k][c] meet at PE(r,c).
   logic signed [DATA_WIDTH-1:0] a_skew [0:SIZE-1][0:SIZE-2];
   logic signed [DATA_WIDTH-1:0] b_skew [0:SIZE-1][0:SIZE-2];
 
   genvar r, c;
   generate
       for (r = 0; r < SIZE; r++) begin : a_skew_gen
           if (r == 0) begin : no_skew_a
               assign a_bus[0][0] = a_gated[0];
           end else begin : has_skew_a
               always_ff @(posedge clk) begin
                   if (rst)
                       a_skew[r][0] <= '0;
                   else
                       a_skew[r][0] <= a_gated[r];
               end
               for (genvar k = 1; k < r; k++) begin : a_chain
                   always_ff @(posedge clk) begin
                       if (rst)
                           a_skew[r][k] <= '0;
                       else
                           a_skew[r][k] <= a_skew[r][k-1];
                   end
               end
               assign a_bus[r][0] = a_skew[r][r-1];
           end
       end
 
       for (c = 0; c < SIZE; c++) begin : b_skew_gen
           if (c == 0) begin : no_skew_b
               assign b_bus[0][0] = b_gated[0];
           end else begin : has_skew_b
               always_ff @(posedge clk) begin
                   if (rst)
                       b_skew[c][0] <= '0;
                   else
                       b_skew[c][0] <= b_gated[c];
               end
               for (genvar k = 1; k < c; k++) begin : b_chain
                   always_ff @(posedge clk) begin
                       if (rst)
                           b_skew[c][k] <= '0;
                       else
                           b_skew[c][k] <= b_skew[c][k-1];
                   end
               end
               assign b_bus[0][c] = b_skew[c][c-1];
           end
       end
   endgenerate
 
   // --- PE array -------------------------------------------------------
   generate
       for (r = 0; r < SIZE; r++) begin : gen_row
           for (c = 0; c < SIZE; c++) begin : gen_col
               pe #(
                   .DATA_WIDTH(DATA_WIDTH),
                   .ACC_WIDTH (ACC_WIDTH)
               ) pe_inst (
                   .clk    (clk),
                   .rst    (rst),
                   .a_in   (a_bus[r][c]),
                   .a_out  (a_bus[r][c+1]),
                   .b_in   (b_bus[r][c]),
                   .b_out  (b_bus[r+1][c]),
                   .acc_out(acc_out[(r*SIZE+c+1)*ACC_WIDTH-1 -: ACC_WIDTH])
               );
           end
       end
   endgenerate
 
endmodule
 
// =============================================================
// pe — Processing element (MAC unit)
// Output-stationary systolic array cell.
// =============================================================
module pe #(
   parameter int DATA_WIDTH = 16,
   parameter int ACC_WIDTH  = 32
)(
   input  logic                          clk,
   input  logic                          rst,
   input  logic signed [DATA_WIDTH-1:0]  a_in,
   output logic signed [DATA_WIDTH-1:0]  a_out,
   input  logic signed [DATA_WIDTH-1:0]  b_in,
   output logic signed [DATA_WIDTH-1:0]  b_out,
   output logic signed [ACC_WIDTH-1:0]   acc_out
);
 
   logic signed [DATA_WIDTH-1:0]   a_reg;
   logic signed [DATA_WIDTH-1:0]   b_reg;
   logic signed [ACC_WIDTH-1:0]    acc_reg;
 
   logic signed [2*DATA_WIDTH-1:0] product;
   assign product = a_in * b_in;
 
   always_ff @(posedge clk) begin
       if (rst) begin
           a_reg   <= '0;
           b_reg   <= '0;
           acc_reg <= '0;
       end else begin
           a_reg   <= a_in;
           b_reg   <= b_in;
           acc_reg <= acc_reg + ACC_WIDTH'(product);
       end
   end
 
   assign a_out   = a_reg;
   assign b_out   = b_reg;
   assign acc_out = acc_reg;
 
endmodule