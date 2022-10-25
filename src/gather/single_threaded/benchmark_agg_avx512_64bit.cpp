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

#include "gather/simd_variants/avx512/agg_avx512_64BitVariants.h"

#define ITERATIONS 10


using namespace std;

struct measures {
    uint64_t result; 
    double duration;
    double throughput;
    double mis;
} scalar, linear, gather, seti, indexed; 


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

bool benchmark(measures* res, uint64_t correct_result, const uint64_t* values, uint64_t n, const uint32_t stride, double GB, function<uint64_t(const uint64_t*,uint64_t, const uint32_t)> func) {

    uint64_t duration = 0;
    for (int i=0; i<ITERATIONS; i++) {
        // flush all caches and TLB 
        // clean start setting
        void flush_cache_all(void);
        void flush_tlb_all(void);
        auto begin = chrono::high_resolution_clock::now();
        (*res).result =  func(values, n, stride);
        auto end = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    }
    (*res).duration = (double)duration/(double)ITERATIONS;
    (*res).throughput = GB/((double)(*res).duration*1e-9);
     (*res).mis = (n/1000000)/((double)duration/(double)((uint64_t)ITERATIONS*(uint64_t)1000000000));
    if ((*res).result == correct_result) return true;
    else return false;
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
    double GB = (((double)number_of_values*sizeof(uint64_t)/(double)1024)/(double)1024)/(double)1024;
  
    /**
     * allocate memory and fill with random numbers
     */
    uint64_t *array_64;
    array_64 = (uint64_t *) aligned_alloc(64, number_of_values * sizeof (uint64_t));
    if (array_64 != NULL) {
        cout << "Memory allocated - " << number_of_values << " values" << endl;
    } else {
        cout << "Memory not allocated" << endl;
    }
    generate_random_values(array_64, number_of_values);
    uint64_t correct = aggregate_scalar(array_64, number_of_values);
    cout <<"Generation done."<<endl; 

    /**
     * run several benchmarks on generated data
     */

    // open files to store runtime measurements
     ofstream result_file;
    result_file.open("./data/512_64bits/results.dat");

    // scalar variant
    if (benchmark(&scalar, correct, array_64, number_of_values, 0, GB, &aggregate_scalar)) {
        cout <<"Scalar done"<<endl;
    }
    else {
        cout <<"scalar failed"<<endl;
    }

    // avx512 linear load variant
     if (benchmark(&linear, correct, array_64, number_of_values,0, GB, &aggregate_linear_avx512)) {
        cout <<"Linear AVX512 done"<<endl;
    }
    else {
        cout <<"Linear AVX512 failed"<<endl;
    }

    for (int stride_pow = 2; stride_pow <= max_stride; stride_pow++) {
        uint64_t stride_size = pow(2, stride_pow);

        // strided access
        if (benchmark(&gather, correct, array_64, number_of_values, stride_size, GB, &aggregate_strided_gather_avx512)) {
            cout <<"Gather - Stride with Size "<<stride_size<<" done"<<endl;
        }
        else {
            cout <<"Gather - Stride with Size "<<stride_size<<" failed"<<endl;
        }

        // set instruction (strided access)
         if (benchmark(&seti, correct, array_64, number_of_values, stride_size, GB, &aggregate_strided_set_avx512)) {
            cout <<"Set - Stride with Size "<<stride_size<<" done"<<endl;
        }
        else {
            cout <<"Set - Stride with Size "<<stride_size<<" failed"<<endl;
        }
        

        // writing results to file
        result_file << stride_size << " " << stride_size * 8 << " "<<scalar.mis<<" "<<
                                                                     scalar.throughput<<" "<<
                                                                     linear.mis<<" "<<
                                                                     linear.throughput<<" "<<
                                                                     gather.mis<<" "<<
                                                                     gather.throughput<<" "<< 
                                                                     seti.mis<<" "<<
                                                                     seti.throughput<<endl;
    }

   

    result_file.close();
    free(array_64);

  

  return 0;
}

