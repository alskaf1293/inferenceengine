// ======================================================================
// tb_supervised_sequence_file_trace4.sv
//
// Persistent four-layer trace for deeper RTL/Python alignment.
// DATA rows are ordered by update:
//   x_top[0..K3-1] y_bottom[0..K0-1]
// ======================================================================

`timescale 1ns/1ps
`default_nettype none

import "DPI-C" function int unsigned real_to_f32 (real r);
import "DPI-C" function real        f32_to_real (int unsigned bits);

module tb_supervised_sequence_file_trace4 #(
  parameter int K0 = 1,
  parameter int K1 = 16,
  parameter int K2 = 16,
  parameter int K3 = 23,
  parameter int UPDATES = 2,
  parameter int SAMPLES_PER_UPDATE = 6,
  parameter int MAX_K = 32,
  parameter int MAX_TOTAL_SAMPLES = 256
);
  `include "tb/tb_logger.sv"

  localparam int NUM_LAYERS = 4;
  localparam int K_LUT[NUM_LAYERS] = '{K0, K1, K2, K3};
  localparam act_kind_e ACT_LUT[NUM_LAYERS] = '{ACT_LINEAR, ACT_RELU, ACT_RELU, ACT_LINEAR};
  localparam int TOTAL_SAMPLES = UPDATES * SAMPLES_PER_UPDATE;

  real X[0:MAX_TOTAL_SAMPLES-1][0:MAX_K-1];
  real Y[0:MAX_TOTAL_SAMPLES-1][0:MAX_K-1];

  int INFER_TICKS, LEARN_TICKS, EVAL_TICKS;
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

  task automatic load_dataset(input string data_path);
    int fd, code;
    real value;
    if (TOTAL_SAMPLES > MAX_TOTAL_SAMPLES) $fatal(1, "[TB] total samples exceeds MAX_TOTAL_SAMPLES");
    fd = $fopen(data_path, "r");
    if (fd == 0) $fatal(1, "[TB] could not open DATA=%s", data_path);
    for (int s = 0; s < TOTAL_SAMPLES; s++) begin
      for (int j = 0; j < K3; j++) begin
        code = $fscanf(fd, "%f", value);
        if (code != 1) $fatal(1, "[TB] malformed x value sample=%0d j=%0d", s, j);
        X[s][j] = value;
      end
      for (int i = 0; i < K0; i++) begin
        code = $fscanf(fd, "%f", value);
        if (code != 1) $fatal(1, "[TB] malformed y value sample=%0d i=%0d", s, i);
        Y[s][i] = value;
      end
    end
    $fclose(fd);
  endtask

  task automatic clear_clamps();
    for (int l = 0; l < NUM_LAYERS; l++) begin
      for (int i = 0; i < MAX_K; i++) begin
        x_set_en_all[l][i] = 1'b0;
        x_obs_flat_all[l][32*i +: 32] = 32'h0000_0000;
      end
    end
  endtask

  task automatic clamp_sample(input int s, input bit clamp_bottom);
    clear_clamps();
    for (int i = 0; i < K3; i++) begin
      x_set_en_all[3][i] = 1'b1;
      x_obs_flat_all[3][32*i +: 32] = f2b(X[s][i]);
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

  task automatic eval_update(input int update_idx, output real mse);
    real acc;
    int start_idx;
    acc = 0.0;
    start_idx = update_idx * SAMPLES_PER_UPDATE;
    set_rates(0.0, GAMMA_R);
    for (int s = 0; s < SAMPLES_PER_UPDATE; s++) begin
      clamp_sample(start_idx + s, 1'b0);
      run_ticks(EVAL_TICKS);
      acc += mse_sample(start_idx + s);
    end
    mse = acc / real'(SAMPLES_PER_UPDATE);
  endtask

  task automatic train_update(input int update_idx);
    int start_idx;
    start_idx = update_idx * SAMPLES_PER_UPDATE;
    for (int s = 0; s < SAMPLES_PER_UPDATE; s++) begin
      clamp_sample(start_idx + s, 1'b1);
      set_rates(0.0, GAMMA_R);
      run_ticks(INFER_TICKS);
      set_rates(ALPHA_R, GAMMA_R);
      run_ticks(LEARN_TICKS);
    end
  endtask

  initial begin
    string data_path, out_path;
    int ok_int;
    real ok_real, mse;

    data_path = "runs/four_layer_sequence.dat";
    out_path = "runs/four_layer_sequence.csv";
    void'($value$plusargs("DATA=%s", data_path));
    void'($value$plusargs("CSV=%s", out_path));
    INFER_TICKS = 8;
    LEARN_TICKS = 2;
    EVAL_TICKS = 10;
    ALPHA_R = 0.02;
    GAMMA_R = 0.05;
    ok_int = 0; if ($value$plusargs("INFER_TICKS=%d", ok_int)) INFER_TICKS = ok_int;
    ok_int = 0; if ($value$plusargs("LEARN_TICKS=%d", ok_int)) LEARN_TICKS = ok_int;
    ok_int = 0; if ($value$plusargs("EVAL_TICKS=%d", ok_int)) EVAL_TICKS = ok_int;
    ok_real = 0.0; if ($value$plusargs("ALPHA=%f", ok_real)) ALPHA_R = ok_real;
    ok_real = 0.0; if ($value$plusargs("GAMMA=%f", ok_real)) GAMMA_R = ok_real;

    start_tick = 1'b0;
    clear_clamps();
    load_dataset(data_path);

    wait(rst_n);
    repeat (5) @(posedge clk);

    csv_open(out_path, "update,mse");
    eval_update(0, mse);
    csv_row($sformatf("%0d,%0.9f", -1, mse));

    for (int u = 0; u < UPDATES; u++) begin
      train_update(u);
      eval_update(u, mse);
      csv_row($sformatf("%0d,%0.9f", u, mse));
    end

    csv_close();
    $finish;
  end
endmodule

`default_nettype wire
