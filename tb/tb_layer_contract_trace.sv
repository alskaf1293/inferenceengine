// ======================================================================
// tb_layer_contract_trace.sv
//
// Parameterized pc_layer tile contract.  This verifies a small parallel
// tile of neurons with large fan-in/fan-out without elaborating a whole
// 256-wide network.
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

import "DPI-C" function int unsigned real_to_f32 (real r);
import "DPI-C" function real        f32_to_real (int unsigned bits);

module tb_layer_contract_trace #(
  parameter int K = 4,
  parameter int N = 256,
  parameter int M = 1,
  parameter int ACT_ID = 0
);
  `include "tb/tb_logger.sv"

  localparam int EXP = 8;
  localparam int SIG = 24;
  localparam int LAST_K = K - 1;
  localparam int LAST_N = N - 1;
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

  function automatic real x_value(input int j);
    x_value = 0.01 * real'((j % 23) - 11);
  endfunction

  function automatic real back_value(input int r, input int i);
    back_value = 0.002 * real'(((r * 7 + i * 3) % 19) - 9);
  endfunction

  logic start_tick, busy_o, done_o, back_valid_o;
  logic [31:0] alpha_ieee, gamma_ieee;
  logic [K-1:0] x_set_en_vec;
  logic [31:0] x_obs_ieee_vec [K];
  logic [31:0] x_up_ieee [N];
  logic [31:0] back_from_down_ieee [M][K];
  logic [31:0] back_matrix_kn_ieee [K][N];
  logic [31:0] back_matrix_nk_ieee [N][K];
  logic [31:0] x_state_ieee [K];

  pc_layer #(
    .K(K),
    .N(N),
    .M(M),
    .EXP(EXP),
    .SIG(SIG),
    .CLAMP_HARD_THIS_LAYER(1'b0),
    .ACT_THIS_LAYER(ACT_KIND_PARAM),
    .X_INIT_IEEE_THIS_LAYER(32'h00000000)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start_tick(start_tick),
    .busy_o(busy_o),
    .done_o(done_o),
    .alpha_ieee(alpha_ieee),
    .gamma_ieee(gamma_ieee),
    .x_set_en_vec(x_set_en_vec),
    .x_obs_ieee_vec(x_obs_ieee_vec),
    .x_up_ieee(x_up_ieee),
    .back_from_down_ieee(back_from_down_ieee),
    .back_up_valid_o(back_valid_o),
    .back_matrix_kn_ieee(back_matrix_kn_ieee),
    .back_matrix_nk_ieee(back_matrix_nk_ieee),
    .x_state_ieee(x_state_ieee)
  );

  logic [31:0] theta00_ieee, theta_last_ieee, bias0_ieee, bias_last_ieee;
  hf_rec2f32 C_T00 (.in_rec(dut.G_NEURON[0].u_core.theta[0]), .out_ieee(theta00_ieee));
  hf_rec2f32 C_TLL (.in_rec(dut.G_NEURON[LAST_K].u_core.theta[LAST_N]), .out_ieee(theta_last_ieee));
  hf_rec2f32 C_B0  (.in_rec(dut.G_NEURON[0].u_core.theta[N]), .out_ieee(bias0_ieee));
  hf_rec2f32 C_BL  (.in_rec(dut.G_NEURON[LAST_K].u_core.theta[N]), .out_ieee(bias_last_ieee));

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
      if (guard > 30000) $fatal(1, "[TB] tick deadlock");
    end
    @(posedge clk);
  endtask

  task automatic write_trace(input int tick_idx);
    csv_row($sformatf(
      "%0d,%0d,%0d,%0d,%0d,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f,%0.9f",
      tick_idx,
      K,
      N,
      M,
      ACT_ID,
      b2f(x_state_ieee[0]),
      b2f(x_state_ieee[LAST_K]),
      b2f(back_matrix_kn_ieee[0][0]),
      b2f(back_matrix_kn_ieee[0][LAST_N]),
      b2f(back_matrix_kn_ieee[LAST_K][0]),
      b2f(back_matrix_kn_ieee[LAST_K][LAST_N]),
      b2f(back_matrix_nk_ieee[0][0]),
      b2f(back_matrix_nk_ieee[LAST_N][LAST_K]),
      b2f(theta00_ieee),
      b2f(theta_last_ieee),
      b2f(bias0_ieee),
      b2f(bias_last_ieee)
    ));
  endtask

  initial begin
    string csv_path;
    csv_path = "runs/layer_contract_trace.csv";
    void'($value$plusargs("CSV=%s", csv_path));

    start_tick = 1'b0;
    alpha_ieee = f2b(0.05);
    gamma_ieee = f2b(0.10);
    x_set_en_vec = '0;
    for (int i = 0; i < K; i++) begin
      x_obs_ieee_vec[i] = f2b(0.0);
    end
    for (int j = 0; j < N; j++) begin
      x_up_ieee[j] = f2b(x_value(j));
    end
    for (int r = 0; r < M; r++) begin
      for (int i = 0; i < K; i++) begin
        back_from_down_ieee[r][i] = f2b(back_value(r, i));
      end
    end

    wait(rst_n);
    repeat (5) @(posedge clk);

    csv_open(csv_path, "tick,K,N,M,act_id,x0,x_last,back00,back0_last,backlast0,backlast_last,backnk00,backnk_last_last,theta00,theta_last,bias0,bias_last");
    do_tick();
    write_trace(1);

    for (int j = 0; j < N; j++) begin
      x_up_ieee[j] = f2b(-0.5 * x_value(j));
    end
    for (int r = 0; r < M; r++) begin
      for (int i = 0; i < K; i++) begin
        back_from_down_ieee[r][i] = f2b(-0.25 * back_value(r, i));
      end
    end
    do_tick();
    write_trace(2);

    csv_close();
    $finish;
  end
endmodule

`default_nettype wire
