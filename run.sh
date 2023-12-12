
# roughly 20 to 30, too low causes segfaults without explanation
# what too low means depends on threadedness and max_stride (hardcoded)
# 2**data_size values are allocated
data_size=26
data_size=$1

# options: single, multi
threadedness=multi
threadedness=$2

# options: avx, avx512
avx_type=avx
avx_type=$3

# options: 64, 32
integer_bitcount=64
integer_bitcount=$4

benchmark=${threadedness}_threaded_benchmark_agg_${avx_type}_${integer_bitcount}

echo "makeing sure, the benchmark $benchmark is compiled up to date"
make $benchmark

logfilename="./data/log_${data_size}_${benchmark}_$(date +%s)"
echo "start time: $(date +'%F_%T')" >> $logfilename
echo "running the benchmark, logging to: $logfilename"
./bin/$benchmark $data_size |& tee --append $logfilename
echo "stop time: $(date +'%F_%T')" >> $logfilename

