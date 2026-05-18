// ======================================================================
// tb_neuron_contract_trace.sv
//
// Parameterized neuron contract trace for large fan-in/fan-out checks.
// This verifies the scaling-sensitive primitive without instantiating a
// full 256-wide network.
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

import "DPI-C" function int unsigned real_to_f32 (real r);
import "DPI-C" function real        f32_to_real (int unsigned bits);

module tb_neuron_contract_trace #(
  parameter int N = 256,
  parameter int M = 1,
  parameter int ACT_ID = 0,
  parameter int CLAMP_HARD_PARAM = 0
);
  `include "tb/tb_logger.sv"

  localparam int EXP = 8;
  localparam int SIG = 24;
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

  function automatic real theta_value(input int j);
    theta_value = 0.001 * real'((j % 17) - 8);
  endfunction

  function automatic real x_value(input int j);
    x_value = 0.01 * real'((j % 23) - 11);
  endfunction

  function automatic real back_value(input int j);
    back_value = 0.002 * real'((j % 19) - 9);
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

  logic [31:0] theta0_ieee, theta_last_ieee, bias_ieee, eps_ieee, mu_ieee, backsum_ieee, x_stored_ieee;
  hf_rec2f32 C_T0 (.in_rec(dut.theta[0]), .out_ieee(theta0_ieee));
  hf_rec2f32 C_TL (.in_rec(dut.theta[N-1]), .out_ieee(theta_last_ieee));
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
      if (guard > 20000) $fatal(1, "[TB] tick deadlock");
    end
    @(posedge clk);
  endtask

  task automatic write_trace(input int tick_idx);
    csv_row($sformatf(
      "%0d,%0d,%0d,%0d,%0d,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f",
      tick_idx,
      N,
      M,
      ACT_ID,
      CLAMP_HARD_PARAM,
      b2f(mu_ieee),
      b2f(eps_ieee),
      b2f(backsum_ieee),
      b2f(back_vec_ieee[0]),
      b2f(back_vec_ieee[N-1]),
      b2f(theta0_ieee),
      b2f(theta_last_ieee),
      b2f(bias_ieee),
      b2f(x_stored_ieee),
      b2f(x_i_ieee),
      b2f(x_obs_ieee),
      b2f(back_in_ieee[0])
    ));
  endtask

  initial begin
    string csv_path;
    int ok_int;
    csv_path = "runs/neuron_contract_trace.csv";
    void'($value$plusargs("CSV=%s", csv_path));

    start_tick = 1'b0;
    x_set_en = 1'b0;
    x_obs_ieee = f2b(0.6);
    alpha_ieee = f2b(0.05);
    gamma_ieee = f2b(0.10);
    ok_int = 0;
    if ($value$plusargs("XSET=%d", ok_int)) x_set_en = (ok_int != 0);

    for (int j = 0; j < N; j++) begin
      theta_preset_ieee[j] = f2b(theta_value(j));
      x_vec_ieee[j] = f2b(x_value(j));
    end
    for (int j = 0; j < M; j++) begin
      back_in_ieee[j] = f2b(back_value(j));
    end

    wait(rst_n);
    repeat (5) @(posedge clk);

    csv_open(csv_path, "tick,N,M,act_id,clamp_hard,mu,eps,backsum,back0,back_last,theta0,theta_last,bias,x_state,x_i,x_obs,back_in0");
    do_tick();
    write_trace(1);

    for (int j = 0; j < N; j++) begin
      x_vec_ieee[j] = f2b(-0.5 * x_value(j));
    end
    for (int j = 0; j < M; j++) begin
      back_in_ieee[j] = f2b(-0.25 * back_value(j));
    end
    do_tick();
    write_trace(2);

    csv_close();
    $finish;
  end
endmodule

`default_nettype wire
