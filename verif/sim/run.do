# Run from verif/sim:   vsim -c -do run.do
#
# Compile order matters for packages (softmax_pkg, tpu_pkg) but not for
# plain modules -- Verilog resolves those at elaboration.

vlib work
vmap work work

# ---- RTL ----------------------------------------------------------------
# softmax_unit.sv defines softmax_pkg, so it goes first.
vlog -sv ../../hardware/modules/softmax_unit.sv
vlog     ../../hardware/modules/uart_rx.v
vlog     ../../hardware/modules/uart_tx.v
vlog -sv ../../hardware/modules/weight_bram.sv
vlog -sv ../../hardware/modules/scratch_bram.sv
vlog -sv ../../hardware/modules/systolic_array_8x8.sv
vlog -sv ../../hardware/modules/tile_controller.sv
vlog -sv ../../hardware/modules/attention_fsm.sv

# ---- Testbench ----------------------------------------------------------
# Questa FSE ships UVM precompiled as library "mtiUvm", which modelsim.ini
# maps to uvm-1.1d. vlog still needs the matching source dir on the include
# path so `include "uvm_macros.svh" resolves.
set UVM_SRC $env(MODEL_TECH)/../verilog_src/uvm-1.1d/src

# +incdir also tells `include where to find the classes in sibling dirs.
vlog -sv ../common/uart_if.sv
vlog -sv ../common/weight_bkdr_if.sv
vlog -sv ../common/softmax_ref_if.sv
vlog -sv +incdir+$UVM_SRC+../agents+../env+../testcases ../tpu_pkg.sv
vlog -sv +incdir+$UVM_SRC ../tb/tb_top.sv

# ---- Elaborate and run --------------------------------------------------
# +acc keeps signal names visible so you can actually see them in the wave.
vopt tb_top -o tb_opt +acc

vsim -L mtiUvm tb_opt +UVM_TESTNAME=tc_uart_rx +UVM_VERBOSITY=UVM_LOW
run -all
