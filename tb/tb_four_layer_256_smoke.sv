// ======================================================================
// tb_four_layer_256_smoke.sv
//
// Minimal 1 -> 256 -> 256 -> 23 four-layer smoke test.
// This intentionally avoids the generic sequence/file harness so the
// generated Verilator model is dominated by pc_network_nlayer itself.
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

import "DPI-C" function int unsigned real_to_f32 (real r);
import "DPI-C" function real        f32_to_real (int unsigned bits);

module tb_four_layer_256_smoke;
  `include "tb/tb_logger.sv"

  localparam int NUM_LAYERS = 4;
  localparam int K0 = 1;
  localparam int K1 = 256;
  localparam int K2 = 256;
  localparam int K3 = 23;
  localparam int MAX_K = 256;
  localparam int K_LUT[NUM_LAYERS] = '{K0, K1, K2, K3};
  localparam act_kind_e ACT_LUT[NUM_LAYERS] = '{ACT_LINEAR, ACT_RELU, ACT_RELU, ACT_LINEAR};

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
    x_top_value = 0.05 * real'(j - 11);
  endfunction

  function automatic real y_bottom_value();
    y_bottom_value = 0.15;
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

  task automatic set_rates(input real alpha_r, input real gamma_r);
    alpha_ieee = f2b(alpha_r);
    gamma_ieee = f2b(gamma_r);
  endtask

  task automatic clear_clamps();
    x_set_en_all = '0;
    x_obs_flat_all = '0;
  endtask

  task automatic clamp_top(input bit clamp_bottom);
    clear_clamps();
    for (int i = 0; i < K3; i++) begin
      x_set_en_all[3][i] = 1'b1;
      x_obs_flat_all[3][32*i +: 32] = f2b(x_top_value(i));
    end
    if (clamp_bottom) begin
      x_set_en_all[0][0] = 1'b1;
      x_obs_flat_all[0][31:0] = f2b(y_bottom_value());
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

  function automatic real x0_state();
    x0_state = b2f(uut.x_state_global[0][0]);
  endfunction

  function automatic real mse();
    real d;
    d = x0_state() - y_bottom_value();
    mse = d * d;
  endfunction

  initial begin
    string out_path;
    real alpha_r, gamma_r;

    out_path = "runs/four_layer_256_smoke.csv";
    void'($value$plusargs("CSV=%s", out_path));
    alpha_r = 0.002;
    gamma_r = 0.05;

    start_tick = 1'b0;
    set_rates(0.0, gamma_r);
    clear_clamps();

    wait(rst_n);
    repeat (5) @(posedge clk);

    csv_open(out_path, "phase,mse,x0");

    clamp_top(1'b0);
    set_rates(0.0, gamma_r);
    do_tick();
    csv_row($sformatf("eval_before,%0.9f,%0.9f", mse(), x0_state()));

    clamp_top(1'b1);
    set_rates(0.0, gamma_r);
    do_tick();

    clamp_top(1'b1);
    set_rates(alpha_r, gamma_r);
    do_tick();

    clamp_top(1'b0);
    set_rates(0.0, gamma_r);
    do_tick();
    csv_row($sformatf("eval_after,%0.9f,%0.9f", mse(), x0_state()));

    csv_close();
    $finish;
  end
endmodule

`default_nettype wire
