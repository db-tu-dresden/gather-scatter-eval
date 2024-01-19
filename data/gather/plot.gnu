set key horizontal
set key off
set xlabel 'stride size [in 2^x number of elements]'
set grid
set term pdf mono font ",22"
set ylabel 'Throughput (GiB/s)'

set output 'avx256_32.pdf'
plot [0:16][0:15]'results_avx256_32bit.dat' using 1:4 w l lw 3 title 'scalar', 'results_avx256_32bit.dat' using 1:5 w l lw 3 title 'linear', 'results_avx256_32bit.dat' using 1:7 w l lw 3 title 'block-strided'

set output 'avx256_64.pdf'
plot [0:16][0:15]'results_avx256_64bit.dat' using 1:4 w l lw 3 title 'scalar', 'results_avx256_64bit.dat' using 1:5 w l lw 3 title 'linear', 'results_avx256_64bit.dat' using 1:7 w l lw 3 title 'block-strided'

set output 'avx512_32.pdf'
plot [0:16][0:15]'results_avx512_32bit.dat' using 1:4 w l lw 3 title 'scalar', 'results_avx512_32bit.dat' using 1:5 w l lw 3 title 'linear', 'results_avx512_32bit.dat' using 1:7 w l lw 3 title 'block-strided'

set output 'avx512_64.pdf'
plot [0:16][0:15]'results_avx512_64bit.dat' using 1:4 w l lw 3 title 'scalar', 'results_avx512_64bit.dat' using 1:5 w l lw 3 title 'linear', 'results_avx512_64bit.dat' using 1:7 w l lw 3 title 'block-strided'