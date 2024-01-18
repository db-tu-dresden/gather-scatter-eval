set key horizontal
set key on
set xlabel 'stride size [in 2^x number of elements]'
set grid
set term pdf mono font ",22"
set ylabel 'Throughput (GiB/s)'

set output 'result.pdf'
plot [][0:30]'results.dat' using 1:4 w l lw 3 title 'scalar', 'results.dat' using 1:6 w l lw 3 title 'linear', 'results.dat' using 1:8 w l lw 3 title 'gather','results.dat' using 1:10 w l lw 3 title 'seti'