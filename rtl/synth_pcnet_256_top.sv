// ======================================================================
// synth_pcnet_256_top.sv
//
// Synthesis/elaboration top for the full HalfCheetah-shaped PC fabric:
//   1 -> 256 -> 256 -> 23
//
// This is not a simulation testbench.  It exists to prove that the full
// fabric structurally elaborates without relying on a Verilator C++ model.
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

module synth_pcnet_256_top (
  input  logic clk,
  input  logic rst_n,
  input  logic start_tick,
  input  logic [31:0] alpha_ieee,
  input  logic [31:0] gamma_ieee,
  output logic busy_o,
  output logic done_o
);
  localparam int NUM_LAYERS = 4;
  localparam int MAX_K = 256;
  localparam int K_LUT[NUM_LAYERS] = '{1, 256, 256, 23};
  localparam act_kind_e ACT_LUT[NUM_LAYERS] = '{ACT_LINEAR, ACT_RELU, ACT_RELU, ACT_LINEAR};

  logic [NUM_LAYERS-1:0][MAX_K-1:0] x_set_en_all;
  logic [NUM_LAYERS-1:0][32*MAX_K-1:0] x_obs_flat_all;

  assign x_set_en_all = '0;
  assign x_obs_flat_all = '0;

  pc_network_nlayer #(
    .NUM_LAYERS(NUM_LAYERS),
    .MAX_K(MAX_K),
    .K_LUT(K_LUT),
    .M0(0),
    .ACT_LUT(ACT_LUT)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start_tick(start_tick),
    .alpha_ieee(alpha_ieee),
    .gamma_ieee(gamma_ieee),
    .x_set_en_all(x_set_en_all),
    .x_obs_flat_all(x_obs_flat_all),
    .busy_o(busy_o),
    .done_o(done_o)
  );
endmodule

`default_nettype wire
