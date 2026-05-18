// ======================================================================
// tb_deep_scheduler_trace.sv
//
// Narrow/deep pc_network_nlayer scheduler contract.  This verifies
// simultaneous tick boundary timing across more layers without width blowup.
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

import "DPI-C" function int unsigned real_to_f32 (real r);
import "DPI-C" function real        f32_to_real (int unsigned bits);

module tb_deep_scheduler_trace #(
  parameter int H = 8
);
  `include "tb/tb_logger.sv"

  localparam int NUM_LAYERS = 5;
  localparam int K0 = 1;
  localparam int K1 = H;
  localparam int K2 = H;
  localparam int K3 = H;
  localparam int K4 = 23;
  localparam int MAX_K = 32;
  localparam int MID_H = H / 2;
  localparam int K_LUT[NUM_LAYERS] = '{K0, K1, K2, K3, K4};
  localparam act_kind_e ACT_LUT[NUM_LAYERS] = '{ACT_LINEAR, ACT_RELU, ACT_RELU, ACT_RELU, ACT_LINEAR};

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

  function automatic real x_top_value(input int j);
    x_top_value = 0.03 * real'((j % 11) - 5);
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

  logic [31:0] l0_w00_ieee, l1_w00_ieee, l2_w00_ieee;
  hf_rec2f32 C_L0W00 (.in_rec(uut.G_LAYER[0].G_LAYER_INST_L0.L.G_NEURON[0].u_core.theta[0]), .out_ieee(l0_w00_ieee));
  hf_rec2f32 C_L1W00 (.in_rec(uut.G_LAYER[1].G_LAYER_INST_L1.L.G_NEURON[0].u_core.theta[0]), .out_ieee(l1_w00_ieee));
  hf_rec2f32 C_L2W00 (.in_rec(uut.G_LAYER[2].G_LAYER_INST_L2.L.G_NEURON[0].u_core.theta[0]), .out_ieee(l2_w00_ieee));

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

  task automatic set_observations(input bit clamp_bottom);
    clear_clamps();
    for (int i = 0; i < K4; i++) begin
      x_set_en_all[4][i] = 1'b1;
      x_obs_flat_all[4][32*i +: 32] = f2b(x_top_value(i));
    end
    if (clamp_bottom) begin
      x_set_en_all[0][0] = 1'b1;
      x_obs_flat_all[0][31:0] = f2b(0.12);
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
      if (guard > 30000) $fatal(1, "[TB] tick deadlock");
    end
    @(posedge clk);
  endtask

  task automatic write_trace(input int tick_idx);
    csv_row($sformatf(
      "%0d,%0d,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f",
      tick_idx,
      H,
      b2f(uut.x_state_global[0][0]),
      b2f(uut.x_state_global[1][0]),
      b2f(uut.x_state_global[1][MID_H]),
      b2f(uut.x_state_global[2][0]),
      b2f(uut.x_state_global[2][MID_H]),
      b2f(uut.x_state_global[3][0]),
      b2f(uut.x_state_global[3][MID_H]),
      b2f(uut.x_state_global[4][0]),
      b2f(uut.back_nk_global[0][0][0]),
      b2f(l0_w00_ieee),
      b2f(l1_w00_ieee),
      b2f(l2_w00_ieee)
    ));
  endtask

  initial begin
    string csv_path;
    csv_path = "runs/deep_scheduler_trace.csv";
    void'($value$plusargs("CSV=%s", csv_path));

    start_tick = 1'b0;
    alpha_ieee = f2b(0.05);
    gamma_ieee = f2b(0.10);
    clear_clamps();

    wait(rst_n);
    repeat (5) @(posedge clk);

    csv_open(csv_path, "tick,H,x0,x1_0,x1_mid,x2_0,x2_mid,x3_0,x3_mid,x4_0,back0_0_0,l0_w00,l1_w00,l2_w00");

    set_observations(1'b1);
    do_tick();
    write_trace(1);

    set_observations(1'b1);
    do_tick();
    write_trace(2);

    set_observations(1'b0);
    alpha_ieee = f2b(0.0);
    do_tick();
    write_trace(3);

    csv_close();
    $finish;
  end
endmodule

`default_nettype wire
