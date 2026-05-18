// ======================================================================
// tb_neuron_tick_trace.sv
//
// Minimal one-neuron trace for RTL/Python dynamics alignment.
// Verifies the primitive equation used by every pc_layer:
//   mu       = sum_j theta_j * phi(x_j) + bias
//   eps      = x_eff - mu
//   back_vec = theta_j * eps
//   theta_j += alpha * eps * phi(x_j)
//   x        = x + gamma * (phi'(x_eff) * sum(back_in) - eps)
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

import "DPI-C" function int unsigned real_to_f32 (real r);
import "DPI-C" function real        f32_to_real (int unsigned bits);

module tb_neuron_tick_trace #(
  parameter int ACT_ID = 0,              // 0 linear, 1 relu, 2 tanh, 3 sigmoid
  parameter int CLAMP_HARD_PARAM = 0
);
  `include "tb/tb_logger.sv"

  localparam int N = 2;
  localparam int M = 2;
  localparam int EXP = 8;
  localparam int SIG = 24;
  localparam int RECW = EXP + SIG + 1;
  localparam act_kind_e ACT_KIND_PARAM =
      (ACT_ID == 1) ? ACT_RELU :
      (ACT_ID == 2) ? ACT_TANH :
      (ACT_ID == 3) ? ACT_SIGMOID :
                      ACT_LINEAR;

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

  logic start_tick, busy_o, done_o, back_valid_o;
  logic [31:0] alpha_ieee, gamma_ieee, x_obs_ieee, x_i_ieee;
  logic x_set_en;
  logic [31:0] theta_preset_ieee [N];
  logic [31:0] x_vec_ieee [N];
  logic [31:0] back_in_ieee [M];
  logic [31:0] back_vec_ieee [N];

  neuron_core_single_back #(
    .CLAMP_HARD(CLAMP_HARD_PARAM != 0),
    .ACT_KIND(ACT_KIND_PARAM),
    .N(N),
    .M(M),
    .EXP(EXP),
    .SIG(SIG),
    .X_I_INIT_IEEE(32'h3E99999A),      // 0.3
    .BIAS_INIT_IEEE(32'h3DCCCCCD),     // 0.1
    .BIAS_LR_SCALE_IEEE(32'h3F800000), // 1.0
    .FREEZE_BIAS(1'b0)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .theta_preset_ieee(theta_preset_ieee),
    .start_tick(start_tick),
    .busy_o(busy_o),
    .done_o(done_o),
    .alpha_ieee(alpha_ieee),
    .gamma_ieee(gamma_ieee),
    .x_set_en(x_set_en),
    .x_obs_ieee(x_obs_ieee),
    .x_vec_ieee(x_vec_ieee),
    .back_in_ieee(back_in_ieee),
    .x_i_ieee(x_i_ieee),
    .back_vec_valid_o(back_valid_o),
    .back_vec_ieee(back_vec_ieee)
  );

  logic [31:0] theta0_ieee, theta1_ieee, bias_ieee, eps_ieee, mu_ieee, backsum_ieee, x_stored_ieee;
  hf_rec2f32 C_T0 (.in_rec(dut.theta[0]), .out_ieee(theta0_ieee));
  hf_rec2f32 C_T1 (.in_rec(dut.theta[1]), .out_ieee(theta1_ieee));
  hf_rec2f32 C_TB (.in_rec(dut.theta[N]), .out_ieee(bias_ieee));
  hf_rec2f32 C_EP (.in_rec(dut.eps_i), .out_ieee(eps_ieee));
  hf_rec2f32 C_MU (.in_rec(dut.mu_acc), .out_ieee(mu_ieee));
  hf_rec2f32 C_BS (.in_rec(dut.back_sum), .out_ieee(backsum_ieee));
  hf_rec2f32 C_XS (.in_rec(dut.x_i), .out_ieee(x_stored_ieee));

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

  task automatic do_tick();
    int guard;
    guard = 0;
    done_edge = 1'b0;
    @(posedge clk) start_tick = 1'b1;
    @(posedge clk) start_tick = 1'b0;
    while (!done_edge) begin
      @(posedge clk);
      guard++;
      if (guard > 256) $fatal(1, "[TB] tick deadlock");
    end
    @(posedge clk);
  endtask

  task automatic write_trace(input int tick_idx);
    csv_row($sformatf(
      "%0d,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f",
      tick_idx,
      b2f(mu_ieee),
      b2f(eps_ieee),
      b2f(backsum_ieee),
      b2f(back_vec_ieee[0]),
      b2f(back_vec_ieee[1]),
      b2f(theta0_ieee),
      b2f(theta1_ieee),
      b2f(bias_ieee),
      b2f(x_stored_ieee),
      b2f(x_obs_ieee)
    ));
  endtask

  initial begin
    string csv_path;
    int ok_int;
    csv_path = "runs/neuron_tick_trace.csv";
    void'($value$plusargs("CSV=%s", csv_path));

    start_tick = 1'b0;
    x_set_en = 1'b0;
    x_obs_ieee = f2b(0.0);
    alpha_ieee = f2b(0.05);
    gamma_ieee = f2b(0.10);
    ok_int = 0;
    if ($value$plusargs("XSET=%d", ok_int)) x_set_en = (ok_int != 0);
    x_obs_ieee = f2b(0.6);

    theta_preset_ieee[0] = f2b(0.25);
    theta_preset_ieee[1] = f2b(-0.40);
    x_vec_ieee[0] = f2b(0.7);
    x_vec_ieee[1] = f2b(-0.2);
    back_in_ieee[0] = f2b(0.15);
    back_in_ieee[1] = f2b(-0.05);

    wait(rst_n);
    repeat (5) @(posedge clk);

    csv_open(csv_path, "tick,mu,eps,backsum,back0,back1,theta0,theta1,bias,x_state,x_obs");
    do_tick();
    write_trace(1);

    x_vec_ieee[0] = f2b(-0.4);
    x_vec_ieee[1] = f2b(0.9);
    back_in_ieee[0] = f2b(-0.10);
    back_in_ieee[1] = f2b(0.05);
    do_tick();
    write_trace(2);

    csv_close();
    $finish;
  end
endmodule

`default_nettype wire
