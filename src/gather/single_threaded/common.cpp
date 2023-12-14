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

#include "gather/simd_variants/avx/agg_avx_64BitVariants.h"
#include "error_codes.h"

// ITERATIONS and MAX_CORES
#include "parameters.h"

using namespace std;

#include "measures.h"
#include "make_label.cpp"

#include "generate_random_values.cpp"
// template <ResultT> bool benchmark(...)
#include "benchmark_single_threaded.cpp"

template <class ResultT>
using aggregation_function_t = uint64_t (*) (
	const ResultT*,
	uint64_t,
	const uint32_t
);

template <class ResultT>
struct aggregator {
	aggregation_function_t<ResultT> function;
	string label;
	bool strided;
};
template <class ResultT>
using aggregator_t = struct aggregator<ResultT>;

template <class ResultT>
int main_single_threaded(
	const vector<aggregator_t<ResultT>> aggregators,
	uint64_t data_size_log2,
	bool multi_threaded,
	bool avx512,
	bool bits64
) {
    // define number of values
    // 27 --> 134 million integers --> 8GB
    // 26 --> 67 million integers --> 4GB
    uint64_t number_of_values = pow(2, data_size_log2);
	cerr << "number_of_values: " << number_of_values << endl;


    // define max stride size (power of 2)
    size_t max_stride = 15;
	cerr << "max_stride: " << max_stride << ", 2**max_stride: " << (1 << max_stride) << endl;

	if (max_stride + 1 < data_size_log2) {
		/* this stride is fine */
	} else {
		cerr
			<< "Data Size is 2**" << data_size_log2 << " == " << (1<<data_size_log2)
			<< " which does not allow the hardcoded maximum stride of "
			<< "2**" << max_stride << " == " << (1<<max_stride) << "!"
		<< endl;
		return DATA_SIZE_TOO_LOW;
	}

    //compute GB for number of values
    double GB = (((double)number_of_values*sizeof(ResultT)/(double)1024)/(double)1024)/(double)1024;


    /**
     * allocate memory and fill with random numbers
     */
    ResultT *array;
    array = (ResultT *) aligned_alloc(8 * sizeof (ResultT), number_of_values * sizeof (ResultT));
    if (array != NULL) {
        cout << "Memory allocated - " << number_of_values << " values" << endl;
    } else {
        cout << "Memory not allocated" << endl;
		exit(NO_MEMORY);
    }
    generate_random_values(array, number_of_values);
    uint64_t correct = aggregate_scalar(array, number_of_values);
    cout <<"Generation done."<<endl;

    /**
     * run several benchmarks on generated data
     */

	// measurement result structs
	vector<struct measures> measurements;
	measurements.assign(aggregators.size(), {0, 0, 0, 0});
	//struct measures scalar, linear, gather, seti, extra_avx512;

    // open files to store runtime measurements
	string result_filename = (
		"./data/gather/"
		+ make_label(data_size_log2, multi_threaded, avx512, bits64)
		+ "_results.dat"
	);
    ofstream result_file;
    result_file.open(result_filename);

	if (result_file.good()) {
		cout << "writing data to '" << result_filename << "'." << endl;
	} else {
		cerr << "writing data to '" << result_filename << "' failed!" << endl;
		return RESULT_FILE_NOT_OPENED;
	}


	// note: the stride is the outer loop for the benefit of the output file,
	// non-strided aggregation methods will still run only once.
	const int min_stride_pow = 1;
	for (int stride_pow = min_stride_pow; stride_pow <= max_stride; stride_pow++) {
		uint64_t stride_size = pow(2, stride_pow);

		result_file
			<< stride_size << " "
			<< stride_size * 8;

		for (int a = 0; a < aggregators.size(); a++) {
			const aggregation_function_t<ResultT>& function = aggregators[a].function;
			const string& label = aggregators[a].label;
			const bool& strided = aggregators[a].strided;

			measures& measurement = measurements[a];

			if (!strided) {
				if (stride_pow == 1) {
					if (benchmark(&measurement, correct, array, number_of_values, 0, GB, function)) {
						cout << label << " done" << endl;
					} else {
						cout << label << " failed" << endl;
					}
				}
			} else {
				if (benchmark(&measurement, correct, array, number_of_values, stride_size, GB, function)) {
					cout << label << " done" << endl;
				} else {
					cout << label << " failed" << endl;
				}
			}

			result_file
				<< " " << measurement.mis
				<< " " << measurement.throughput;
		}

		result_file << endl;
	}
    result_file.close();

	cerr << "freeing array!" << endl;
    free(array);

	return SUCCESS;
}

