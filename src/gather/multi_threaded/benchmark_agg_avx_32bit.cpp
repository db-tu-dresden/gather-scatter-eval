#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<random>
#include<chrono>
#include "immintrin.h"
#include<fstream>
#include <string.h>
#include <math.h> 
#include <functional>
#include <thread>
#include <vector>
#include <future>
#include <map>
#include <algorithm>

#include "gather/simd_variants/avx/agg_avx_32BitVariants.h"

#define ITERATIONS 10
/* Has to be divisible by 2 */
#define MAX_CORES 8

using namespace std;

struct measures {
    uint64_t result; 
    double duration;
    double throughput;
    double mis;
}; 

typedef function<uint64_t(const uint32_t*,uint64_t, const uint32_t)> benchmark_function;

typedef std::map<uint64_t, struct measures> multithreaded_measures;
multithreaded_measures scalar, linear, gather, seti, stream;

template< typename Function >
std::thread* create_thread( const uint64_t tid, uint32_t* local_result, double* local_duration, bool* local_ready, std::shared_future< void >* sync_barrier, Function&& magic, benchmark_function func ) {
    cpu_set_t cpuset;
    CPU_ZERO( &cpuset );
    CPU_SET( tid, &cpuset );
    std::thread* t = new std::thread( std::forward< Function >( magic ), tid, local_result, local_duration, local_ready, sync_barrier, func );
    int rc = pthread_setaffinity_np( t->native_handle(), sizeof( cpu_set_t ), &cpuset );
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        exit( -10 );
    }
    return t;
}

void print_max_tput( std::ostream& log, multithreaded_measures& results ) {
    double curr_max = 0.0;
    size_t max_cores = 0;
    for ( auto it = results.begin(); it != results.end(); ++it ) {
        if ( it->second.throughput > curr_max ) {
            curr_max = it->second.throughput;
            max_cores = it->first;
        }
    }
    log << "Cores: " << max_cores << " TPut: " << curr_max << std::endl;
}

void log_multithreaded_results( std::ostream& logfile, const size_t stride_size, const std::string ident, multithreaded_measures& results ) {
    for ( auto it = results.begin(); it != results.end(); ++it ) {
        logfile << ident << " " << stride_size << " " << stride_size * 8 << " " << it->first << " " << it->second.throughput << std::endl;
    }
}

/* We anticipate the following order: scalar, linear, gather, seti */
void log_multithreaded_results_per_file( std::string basename, const size_t stride_size, std::vector< multithreaded_measures* > results, bool clean ) {
    std::vector< uint64_t > keys;
    /* Extract all keys */
    for ( auto it = results[0]->begin(); it != results[0]->end(); ++it ) {
        keys.push_back( it->first );
    }

    if ( clean ) {
        for ( auto key : keys ) {
            std::string del_file = basename + "_" + std::to_string( key ) + "_cores.dat";
            if ( remove( del_file.c_str() ) == 0 ) {
                std::cout << "Succesfully removed " << del_file << " before the benchmark." << std::endl;
            } else {
                std::cout << "ERROR removing " << del_file << " before the benchmark (maybe file was not present anyway). CHECK RESULTS" << std::endl;
            }
        }
    }

    for ( auto key : keys ) {
        std::ofstream out( basename + "_" + std::to_string( key ) + "_cores.dat", std::ios_base::app );
        out << stride_size << " " << stride_size * 8;
        for ( auto r : results ) {
            out << " " << (*r)[ key ].mis << " " << (*r)[ key ].throughput;
        }
        out << std::endl;
        out.close();
    }
}

void print_multithreaded_results( std::ostream& logfile, std::string ident, multithreaded_measures& results ) {
    for ( auto it = results.begin(); it != results.end(); ++it ) {
        logfile << "[" << ident << "] Core Count: " << it->first << " TPut: " << it->second.throughput << std::endl;
    }
}

template <typename T>
void generate_random_values(T* array, uint64_t number) {
  static_assert(is_integral<T>::value, "Data type is not integral.");
  std::random_device rd;
  std::mt19937::result_type seed = rd() ^ (
          (std::mt19937::result_type)
          std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch()
          ).count() +
          (std::mt19937::result_type)
          std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch()
          ).count());

  std::mt19937 gen(seed);
  std::uniform_int_distribution<T> distrib(1, 6);

  for (uint64_t j = 0; j < number; ++j) {
    array[j] = distrib(gen);
  }
}

bool benchmark(multithreaded_measures* res, uint64_t correct_result, const uint32_t* values, uint64_t n, const uint32_t stride, double GB, benchmark_function func) {
    for ( size_t core_cnt = 1; core_cnt <= MAX_CORES; core_cnt *= 2 ) { /* Run with 1, 2, 4, ... MAX_CORES cores */
        std::vector< std::thread* > pool;
        
        uint32_t* tmp_res = (uint32_t*) aligned_alloc( 32, core_cnt * sizeof( uint32_t) );
        double* tmp_dur   = (double*)   aligned_alloc( 64, core_cnt * sizeof( double )  );
        bool* ready_vec = (bool*) malloc( core_cnt * sizeof( bool ) );

        auto magic = [core_cnt, values, n, stride] ( const uint64_t tid, uint32_t* local_result, double* local_duration, bool* local_ready, std::shared_future< void >* sync_barrier, benchmark_function local_func ) {
            // flush all caches and TLB 
            // clean start setting
            void flush_cache_all(void);
            void flush_tlb_all(void);
            const uint64_t my_value_count = n / core_cnt; /* Should be always divisible by 2, 4 or 8 */
            const uint32_t my_offset = tid * my_value_count;
            local_ready[ tid ] = true;
            sync_barrier->wait();

            auto begin = chrono::high_resolution_clock::now();
            local_result[ tid ] = local_func(values + my_offset, my_value_count, stride);
            auto end = std::chrono::high_resolution_clock::now();

            local_duration[ tid ] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();        
        };

        double averaged_duration = 0.0;
        for (int i=0; i<ITERATIONS; i++) {
            std::promise< void > p;
		    std::shared_future< void > ready_future( p.get_future( ) );

            memset( tmp_res, 0, core_cnt * sizeof( uint32_t ) );
            memset( tmp_dur, 0, core_cnt * sizeof( double ) );
            memset( ready_vec, 0, core_cnt * sizeof( bool ) );

            for ( size_t tid = 0; tid < core_cnt; ++tid ) {
                pool.emplace_back( create_thread( tid, tmp_res, tmp_dur, ready_vec, &ready_future, magic, func ) );
            }
            bool all_ready = false;
            while ( !all_ready ) {
                /* Wait until all threads are ready to go before we pull the trigger */
                using namespace std::chrono_literals;
                std::this_thread::sleep_for( 1ms );
                all_ready = true;
                for ( size_t i = 0; i < core_cnt; ++i ) {
                    all_ready &= ready_vec[ i ];
                }
            }
            p.set_value(); /* Start execution by notifying on the void promise */
            std::for_each( pool.begin(), pool.end(), []( std::thread* t ) { t->join(); delete t; } ); /* Join and delete threads as soon as they are finished */
            pool.clear();
            double iteration_duration = 0.0;
            for ( size_t i = 0; i < core_cnt; ++i ) {
                iteration_duration += tmp_dur[ i ];
            }
            averaged_duration += iteration_duration / static_cast< double >( core_cnt );
        }

        /* Beware, this is an average of averages. We can also do average of max(thread_runtimes) */
        const double cur_dur = static_cast< double >( averaged_duration ) / static_cast< double >( ITERATIONS );
        /* Integer in Millions / time * 10^9 (becausue nanoseconds) */
        const double cur_mis = ( static_cast<double>( n ) / 1000000.0 ) / ( cur_dur * 1e-9 );
        const double cur_tput = GB / ( cur_dur * 1e-9 );
        uint64_t cur_res = 0;
        for ( size_t i = 0; i < core_cnt; ++i ) {
            cur_res += tmp_res[ i ];
        }

        const struct measures tmp_measures = { cur_res, cur_dur, cur_tput, cur_mis };
        (*res)[ core_cnt ] = tmp_measures;

        free( ready_vec );
        free( tmp_dur );
        free( tmp_res );
    }

    bool success = true;
    for ( size_t core_cnt = 1; core_cnt <= MAX_CORES; core_cnt *= 2 ) {
        success &= (*res)[ core_cnt ].result == correct_result;
    }

    return success;
}



int main(int argc, char** argv) {
    
    if (argc < 2) {
        cout <<"Data Size as input expected!"<<endl;
        return 0;
    }

    int p = atoi(argv[1]);

    // define number of values
    // 27 --> 134 million integers --> 8GB
    // 26 --> 67 million integers --> 4GB
    uint64_t number_of_values = pow(2, p);

    
    // define max stride size (power of 2)
    size_t max_stride = 15;

    //compute GB for number of values
    double GB = (((double)number_of_values*sizeof(uint32_t)/(double)1024)/(double)1024)/(double)1024;
  

    /**
     * allocate memory and fill with random numbers
     */
    uint32_t *array_32;
    array_32 = (uint32_t *) aligned_alloc(32, number_of_values * sizeof (uint32_t));
    if (array_32 != NULL) {
        cout << "Memory allocated - " << number_of_values << " values" << endl;
    } else {
        cout << "Memory not allocated" << endl;
    }
    generate_random_values(array_32, number_of_values);
    uint32_t correct = aggregate_scalar(array_32, number_of_values);
    cout <<"Generation done."<<endl; 

    /**
     * run several benchmarks on generated data
     */

    // open files to store runtime measurements
    // ofstream result_file; /* Enable onle for single-file-logging! */
    // result_file.open("./data/256_32bits/results.dat");  /* Enable onle for single-file-logging! */

    // scalar variant
    if (benchmark(&scalar, correct, array_32, number_of_values, 0, GB, &aggregate_scalar)) {
        cout <<"Scalar done ";
        print_max_tput( cout, scalar );
    }
    else {
        cout <<"scalar failed"<<endl;
    }


    // avx512 linear load variant
     if (benchmark(&linear, correct, array_32, number_of_values,0, GB, &aggregate_linear_avx256)) {
        cout <<"Linear AVX512 done ";
        print_max_tput( cout, linear );
    }
    else {
        cout <<"Linear AVX512 failed" << std::endl;
    }

     // avx512 linear stream load variant
     if (benchmark(&stream, correct, array_32, number_of_values,0, GB, &aggregate_stream_linear_avx256)) {
        cout <<"Linear AVX512 done"<<endl;
    }
    else {
        cout <<"Linear AVX512 failed"<<endl;
    }


    // strided access evaluation using different strides
    bool first_run = true;
    for (int stride_pow = 0; stride_pow <= max_stride; stride_pow++) {
        uint64_t stride_size = pow(2, stride_pow);
        // gather instruction
        if (benchmark(&gather, correct, array_32, number_of_values, stride_size, GB, &aggregate_strided_gather_avx256)) {
            cout <<"Gather - Stride with Size "<<stride_size<<" done ";
            print_max_tput( cout, gather );
        }
        else {
            cout <<"Gather - Stride with Size "<<stride_size<<" failed"<<endl;
        }

        // set instruction
         if (benchmark(&seti, correct, array_32, number_of_values, stride_size, GB, &aggregate_strided_set_avx512)) {
            cout <<"Set - Stride with Size "<<stride_size<<" done ";
            print_max_tput( cout, seti );
        }
        else {
            cout <<"Set - Stride with Size "<<stride_size<<" failed"<<endl;
        }
        // writing results to file
        // log_multithreaded_results( result_file, stride_size, "scalar", scalar ); /* Enable onle for single-file-logging! */
        // log_multithreaded_results( result_file, stride_size, "linear", linear ); /* Enable onle for single-file-logging! */
        // log_multithreaded_results( result_file, stride_size, "gather", gather ); /* Enable onle for single-file-logging! */
        // log_multithreaded_results( result_file, stride_size, "seti", seti ); /* Enable onle for single-file-logging! */
        log_multithreaded_results_per_file( "./data/results", stride_pow, { &scalar, &linear, &gather, &seti }, first_run );
        if ( first_run ) {
            first_run = false;
        }
    }
    print_multithreaded_results( cout, "scalar", scalar );
    print_multithreaded_results( cout, "linear", linear );
    print_multithreaded_results( cout, "gather", gather );
    print_multithreaded_results( cout, "seti", seti );

    // result_file.close(); /* Enable onle for single-file-logging! */
    free(array_32);

    return 0;
}