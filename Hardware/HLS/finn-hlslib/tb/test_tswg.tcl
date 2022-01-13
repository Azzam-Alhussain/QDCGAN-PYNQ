
open_project hls-syn-tswg
add_files input_tswg.cpp -cflags "-std=c++0x -I$::env(FINN_HLS_ROOT)" 
add_files -tb tswg_tb.cpp -cflags "-std=c++0x -I$::env(FINN_HLS_ROOT)" 
set_top Testbench
open_solution sol1
set_part {xczu3eg-sbva484-1-i}
create_clock -period 5 -name default
csim_design
csynth_design
cosim_design
exit
