cmake_minimum_required(VERSION 3.1)

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_FLAGS "-static -Wl,--whole-archive -lpthread -lnuma -Wl,--no-whole-archive")

project(GATHER-SCATTER-EVAL)
set(CMAKE_BINARY_DIR "bin")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_compile_options(-O3 -fno-tree-vectorize -march=sapphirerapids -mavx2 -mavx -mavx512f -mavx512cd)
#add_compile_options(-O3 -fno-tree-vectorize -mavx2 -mavx)

# single threaded benchmarks
add_executable(single_threaded_benchmark_agg_avx_32 src/gather/single_threaded/benchmark_agg_avx_32bit.cpp)
add_executable(single_threaded_benchmark_agg_avx_64 src/gather/single_threaded/benchmark_agg_avx_64bit.cpp)
add_executable(single_threaded_benchmark_agg_avx512_32 src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp)
add_executable(single_threaded_benchmark_agg_avx512_64 src/gather/single_threaded/benchmark_agg_avx512_64bit.cpp)
target_include_directories(single_threaded_benchmark_agg_avx_32 PRIVATE include/)
target_include_directories(single_threaded_benchmark_agg_avx_64 PRIVATE include/)
target_include_directories(single_threaded_benchmark_agg_avx512_32 PRIVATE include/)
target_include_directories(single_threaded_benchmark_agg_avx512_64 PRIVATE include/)

# multi-threaded benchmarks
add_executable(multi_threaded_benchmark_agg_avx_32 src/gather/multi_threaded/benchmark_agg_avx_32bit.cpp)
add_executable(multi_threaded_benchmark_agg_avx_64 src/gather/multi_threaded/benchmark_agg_avx_64bit.cpp)
add_executable(multi_threaded_benchmark_agg_avx512_32 src/gather/multi_threaded/benchmark_agg_avx512_32bit.cpp)
add_executable(multi_threaded_benchmark_agg_avx512_64 src/gather/multi_threaded/benchmark_agg_avx512_64bit.cpp)
target_include_directories(multi_threaded_benchmark_agg_avx_32 PRIVATE include/)
target_include_directories(multi_threaded_benchmark_agg_avx_64 PRIVATE include/)
target_include_directories(multi_threaded_benchmark_agg_avx512_32 PRIVATE include/)
target_include_directories(multi_threaded_benchmark_agg_avx512_64 PRIVATE include/)

TARGET_LINK_LIBRARIES(multi_threaded_benchmark_agg_avx_32
    pthread
)

TARGET_LINK_LIBRARIES(multi_threaded_benchmark_agg_avx_64
    pthread
)

TARGET_LINK_LIBRARIES(multi_threaded_benchmark_agg_avx512_32
    pthread
)

TARGET_LINK_LIBRARIES(multi_threaded_benchmark_agg_avx512_64
    pthread
)