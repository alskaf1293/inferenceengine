// ======================================================================
// tb_network_tick_trace.sv
//
// Deterministic pc_network_nlayer trace for RTL/Python alignment.
// Uses the same 3-layer 2 -> 4 -> 3 shape as the small supervised tests,
// but logs internal layer states and representative weights every tick.
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

import "DPI-C" function int unsigned real_to_f32 (real r);
import "DPI-C" function real        f32_to_real (int unsigned bits);

module tb_network_tick_trace #(
  parameter int ACT0_ID = 0,
  parameter int ACT1_ID = 1,
  parameter int ACT2_ID = 0
);
  `include "tb/tb_logger.sv"

  localparam int NUM_LAYERS = 3;
  localparam int K0 = 3;
  localparam int K1 = 4;
  localparam int K2 = 2;
  localparam int MAX_K = 16;
  localparam int K_LUT[NUM_LAYERS] = '{K0, K1, K2};
  localparam act_kind_e ACT0_KIND =
      (ACT0_ID == 1) ? ACT_RELU :
      (ACT0_ID == 2) ? ACT_TANH :
      (ACT0_ID == 3) ? ACT_SIGMOID :
                       ACT_LINEAR;
  localparam act_kind_e ACT1_KIND =
      (ACT1_ID == 1) ? ACT_RELU :
      (ACT1_ID == 2) ? ACT_TANH :
      (ACT1_ID == 3) ? ACT_SIGMOID :
                       ACT_LINEAR;
  localparam act_kind_e ACT2_KIND =
      (ACT2_ID == 1) ? ACT_RELU :
      (ACT2_ID == 2) ? ACT_TANH :
      (ACT2_ID == 3) ? ACT_SIGMOID :
                       ACT_LINEAR;
  localparam act_kind_e ACT_LUT[NUM_LAYERS] = '{ACT0_KIND, ACT1_KIND, ACT2_KIND};

  logic clk, rst_n;
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  initial begin
    rst_n = 1'b0;
    #50 rst_n = 1'b1;
  end

  function automatic [31:0] f2b(input real r);
    f2b = real_to_f32(r);
  endfunction

  function automatic real b2f(input [31:0] b);
    int unsigned u;
    u = b;
    return f32_to_real(u);
  endfunction

  logic start_tick, busy_o, done_o;
  logic [31:0] alpha_ieee, gamma_ieee;
  logic [NUM_LAYERS-1:0][MAX_K-1:0] x_set_en_all;
  logic [NUM_LAYERS-1:0][32*MAX_K-1:0] x_obs_flat_all;

  pc_network_nlayer #(
    .NUM_LAYERS(NUM_LAYERS),
    .MAX_K(MAX_K),
    .K_LUT(K_LUT),
    .M0(0),
    .ACT_LUT(ACT_LUT)
  ) uut (
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

  logic [31:0] l0_w00_ieee, l0_w01_ieee, l0_b0_ieee;
  logic [31:0] l1_w00_ieee, l1_w01_ieee, l1_b0_ieee;
  hf_rec2f32 C_L0W00 (.in_rec(uut.G_LAYER[0].G_LAYER_INST_L0.L.G_NEURON[0].u_core.theta[0]), .out_ieee(l0_w00_ieee));
  hf_rec2f32 C_L0W01 (.in_rec(uut.G_LAYER[0].G_LAYER_INST_L0.L.G_NEURON[0].u_core.theta[1]), .out_ieee(l0_w01_ieee));
  hf_rec2f32 C_L0B0  (.in_rec(uut.G_LAYER[0].G_LAYER_INST_L0.L.G_NEURON[0].u_core.theta[K1]), .out_ieee(l0_b0_ieee));
  hf_rec2f32 C_L1W00 (.in_rec(uut.G_LAYER[1].G_LAYER_INST_L1.L.G_NEURON[0].u_core.theta[0]), .out_ieee(l1_w00_ieee));
  hf_rec2f32 C_L1W01 (.in_rec(uut.G_LAYER[1].G_LAYER_INST_L1.L.G_NEURON[0].u_core.theta[1]), .out_ieee(l1_w01_ieee));
  hf_rec2f32 C_L1B0  (.in_rec(uut.G_LAYER[1].G_LAYER_INST_L1.L.G_NEURON[0].u_core.theta[K2]), .out_ieee(l1_b0_ieee));

  logic done_q, done_edge;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done_q <= 1'b0;
      done_edge <= 1'b0;
    end else begin
      done_q <= done_o;
      if (done_o & ~done_q) done_edge <= 1'b1;
    end
  end

  task automatic clear_clamps();
    x_set_en_all = '0;
    x_obs_flat_all = '0;
  endtask

  task automatic set_top_bottom(input real x0, input real x1,
                                input real y0, input real y1, input real y2,
                                input bit clamp_bottom);
    clear_clamps();
    x_set_en_all[2][0] = 1'b1;
    x_set_en_all[2][1] = 1'b1;
    x_obs_flat_all[2][32*0 +: 32] = f2b(x0);
    x_obs_flat_all[2][32*1 +: 32] = f2b(x1);
    if (clamp_bottom) begin
      x_set_en_all[0][0] = 1'b1;
      x_set_en_all[0][1] = 1'b1;
      x_set_en_all[0][2] = 1'b1;
      x_obs_flat_all[0][32*0 +: 32] = f2b(y0);
      x_obs_flat_all[0][32*1 +: 32] = f2b(y1);
      x_obs_flat_all[0][32*2 +: 32] = f2b(y2);
    end
  endtask

  task automatic do_tick();
    int guard;
    guard = 0;
    done_edge = 1'b0;
    @(posedge clk) start_tick = 1'b1;
    @(posedge clk) start_tick = 1'b0;
    while (!done_edge) begin
      @(posedge clk);
      guard++;
      if (guard > 8192) $fatal(1, "[TB] tick deadlock");
    end
    @(posedge clk);
  endtask

  task automatic write_trace(input int tick_idx);
    csv_row($sformatf(
      "%0d,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f",
      tick_idx,
      b2f(uut.x_state_global[0][0]),
      b2f(uut.x_state_global[0][1]),
      b2f(uut.x_state_global[0][2]),
      b2f(uut.x_state_global[1][0]),
      b2f(uut.x_state_global[1][1]),
      b2f(uut.x_state_global[1][2]),
      b2f(uut.x_state_global[1][3]),
      b2f(uut.x_state_global[2][0]),
      b2f(uut.x_state_global[2][1]),
      b2f(l0_w00_ieee),
      b2f(l0_w01_ieee),
      b2f(l0_b0_ieee),
      b2f(l1_w00_ieee),
      b2f(l1_w01_ieee),
      b2f(l1_b0_ieee),
      b2f(uut.back_nk_global[0][0][0])
    ));
  endtask

  initial begin
    string csv_path;
    real alpha_r, gamma_r;
    csv_path = "runs/network_tick_trace.csv";
    void'($value$plusargs("CSV=%s", csv_path));
    alpha_r = 0.05;
    gamma_r = 0.10;
    void'($value$plusargs("ALPHA=%f", alpha_r));
    void'($value$plusargs("GAMMA=%f", gamma_r));

    start_tick = 1'b0;
    alpha_ieee = f2b(alpha_r);
    gamma_ieee = f2b(gamma_r);
    clear_clamps();

    wait(rst_n);
    repeat (5) @(posedge clk);

    csv_open(csv_path, "tick,x0_0,x0_1,x0_2,x1_0,x1_1,x1_2,x1_3,x2_0,x2_1,l0_w00,l0_w01,l0_b0,l1_w00,l1_w01,l1_b0,l0_back_0_0");

    set_top_bottom(0.7, -0.2, 0.1, -0.3, 0.2, 1'b1);
    do_tick();
    write_trace(1);

    set_top_bottom(0.7, -0.2, 0.1, -0.3, 0.2, 1'b1);
    do_tick();
    write_trace(2);

    set_top_bottom(-0.4, 0.9, -0.2, 0.05, 0.3, 1'b1);
    do_tick();
    write_trace(3);

    set_top_bottom(-0.4, 0.9, -0.2, 0.05, 0.3, 1'b0);
    alpha_ieee = f2b(0.0);
    do_tick();
    write_trace(4);

    csv_close();
    $finish;
  end
endmodule

`default_nettype wire
