// ======================================================================
// tb_supervised_grid_trace.sv
//
// Parameterized deterministic supervised learning trace for RTL/Python
// alignment. Supports larger 3-layer configs within theta_init_pkg bounds.
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

import "DPI-C" function int unsigned real_to_f32 (real r);
import "DPI-C" function real        f32_to_real (int unsigned bits);

module tb_supervised_grid_trace #(
  parameter int K0 = 3,
  parameter int K1 = 4,
  parameter int K2 = 2,
  parameter int NUM_SAMPLES = 8,
  parameter int MAX_K = 16,
  parameter int MAX_SAMPLES = 64
);
  `include "tb/tb_logger.sv"

  localparam int NUM_LAYERS = 3;
  localparam int K_LUT[NUM_LAYERS] = '{K0, K1, K2};
  localparam act_kind_e ACT_LUT[NUM_LAYERS] = '{ACT_LINEAR, ACT_RELU, ACT_LINEAR};

  real X[0:MAX_SAMPLES-1][0:MAX_K-1];
  real Y[0:MAX_SAMPLES-1][0:MAX_K-1];
  real B_gt[0:MAX_K-1][0:MAX_K-1];
  real A_gt[0:MAX_K-1][0:MAX_K-1];

  int EPOCHS, INFER_TICKS, LEARN_TICKS, EVAL_TICKS;
  real ALPHA_R, GAMMA_R;

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

  function automatic real relu_r(input real z);
    return (z > 0.0) ? z : 0.0;
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

  task automatic build_teacher();
    for (int i = 0; i < MAX_K; i++) begin
      for (int j = 0; j < MAX_K; j++) begin
        B_gt[i][j] = 0.0;
        A_gt[i][j] = 0.0;
      end
    end

    for (int i = 0; i < K1; i++) begin
      int row_type, j0, j1;
      row_type = i % 4;
      j0 = (K2 == 1) ? 0 : (i % K2);
      j1 = (K2 == 1) ? 0 : ((i + 1) % K2);
      case (row_type)
        0: begin B_gt[i][j0] =  1.00; if (K2 > 1) B_gt[i][j1] = -0.20; end
        1: begin B_gt[i][j0] = -0.15; if (K2 > 1) B_gt[i][j1] =  0.95; end
        2: begin B_gt[i][j0] =  0.70; if (K2 > 1) B_gt[i][j1] =  0.25; end
        default: begin B_gt[i][j0] = 0.20; if (K2 > 1) B_gt[i][j1] =  0.80; end
      endcase
    end

    for (int o = 0; o < K0; o++) begin
      int row_type, base;
      row_type = o % 3;
      base = (K1 <= 4) ? 0 : ((2*o) % K1);
      case (row_type)
        0: begin
          if (base + 0 < K1) A_gt[o][base + 0] =  0.90;
          if (base + 1 < K1) A_gt[o][base + 1] = -0.45;
          if (base + 2 < K1) A_gt[o][base + 2] =  0.30;
        end
        1: begin
          if (base + 0 < K1) A_gt[o][base + 0] = -0.70;
          if (base + 1 < K1) A_gt[o][base + 1] =  0.85;
          if (base + 3 < K1) A_gt[o][base + 3] =  0.25;
        end
        default: begin
          if (base + 0 < K1) A_gt[o][base + 0] =  0.50;
          if (base + 1 < K1) A_gt[o][base + 1] =  0.60;
          if (base + 2 < K1) A_gt[o][base + 2] = -0.20;
          if (base + 3 < K1) A_gt[o][base + 3] =  0.35;
        end
      endcase
    end
  endtask

  task automatic build_dataset();
    real h[0:MAX_K-1];
    if (NUM_SAMPLES > MAX_SAMPLES) $fatal(1, "[TB] NUM_SAMPLES exceeds MAX_SAMPLES");
    build_teacher();
    for (int s = 0; s < NUM_SAMPLES; s++) begin
      for (int j = 0; j < K2; j++) begin
        int raw;
        raw = ((s * 17 + j * 11) % 25) - 12;
        X[s][j] = real'(raw) / 10.0;
      end
      for (int i = 0; i < K1; i++) begin
        real acc;
        acc = 0.0;
        for (int j = 0; j < K2; j++) acc += B_gt[i][j] * X[s][j];
        h[i] = relu_r(acc);
      end
      for (int o = 0; o < K0; o++) begin
        real acc;
        acc = 0.0;
        for (int i = 0; i < K1; i++) acc += A_gt[o][i] * h[i];
        Y[s][o] = acc;
      end
    end
  endtask

  task automatic clear_clamps();
    x_set_en_all = '0;
    x_obs_flat_all = '0;
  endtask

  task automatic clamp_sample(input int s, input bit clamp_bottom);
    clear_clamps();
    for (int i = 0; i < K2; i++) begin
      x_set_en_all[2][i] = 1'b1;
      x_obs_flat_all[2][32*i +: 32] = f2b(X[s][i]);
    end
    if (clamp_bottom) begin
      for (int i = 0; i < K0; i++) begin
        x_set_en_all[0][i] = 1'b1;
        x_obs_flat_all[0][32*i +: 32] = f2b(Y[s][i]);
      end
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

  task automatic run_ticks(input int n);
    for (int t = 0; t < n; t++) do_tick();
  endtask

  task automatic set_rates(input real alpha_r, input real gamma_r);
    alpha_ieee = f2b(alpha_r);
    gamma_ieee = f2b(gamma_r);
  endtask

  function automatic real mse_sample(input int s);
    real acc;
    acc = 0.0;
    for (int i = 0; i < K0; i++) begin
      real d;
      d = b2f(uut.x_state_global[0][i]) - Y[s][i];
      acc += d*d;
    end
    return acc / real'(K0);
  endfunction

  task automatic eval_mse(output real mse);
    real acc;
    acc = 0.0;
    set_rates(0.0, GAMMA_R);
    for (int s = 0; s < NUM_SAMPLES; s++) begin
      clamp_sample(s, 1'b0);
      run_ticks(EVAL_TICKS);
      acc += mse_sample(s);
    end
    mse = acc / real'(NUM_SAMPLES);
  endtask

  initial begin
    string out_path;
    int ok_int;
    real ok_real, mse;

    out_path = "runs/supervised_grid_trace.csv";
    void'($value$plusargs("CSV=%s", out_path));
    EPOCHS = 3;
    INFER_TICKS = 10;
    LEARN_TICKS = 2;
    EVAL_TICKS = 20;
    ALPHA_R = 0.05;
    GAMMA_R = 0.10;
    ok_int = 0; if ($value$plusargs("EPOCHS=%d", ok_int)) EPOCHS = ok_int;
    ok_int = 0; if ($value$plusargs("INFER_TICKS=%d", ok_int)) INFER_TICKS = ok_int;
    ok_int = 0; if ($value$plusargs("LEARN_TICKS=%d", ok_int)) LEARN_TICKS = ok_int;
    ok_int = 0; if ($value$plusargs("EVAL_TICKS=%d", ok_int)) EVAL_TICKS = ok_int;
    ok_real = 0.0; if ($value$plusargs("ALPHA=%f", ok_real)) ALPHA_R = ok_real;
    ok_real = 0.0; if ($value$plusargs("GAMMA=%f", ok_real)) GAMMA_R = ok_real;

    start_tick = 1'b0;
    clear_clamps();
    build_dataset();

    wait(rst_n);
    repeat (5) @(posedge clk);

    csv_open(out_path, "epoch,mse");
    eval_mse(mse);
    csv_row($sformatf("%0d,%0.9f", 0, mse));

    for (int ep = 0; ep < EPOCHS; ep++) begin
      for (int s = 0; s < NUM_SAMPLES; s++) begin
        clamp_sample(s, 1'b1);
        set_rates(0.0, GAMMA_R);
        run_ticks(INFER_TICKS);
        set_rates(ALPHA_R, GAMMA_R);
        run_ticks(LEARN_TICKS);
      end
      eval_mse(mse);
      csv_row($sformatf("%0d,%0.9f", ep + 1, mse));
    end

    csv_close();
    $finish;
  end
endmodule

`default_nettype wire
