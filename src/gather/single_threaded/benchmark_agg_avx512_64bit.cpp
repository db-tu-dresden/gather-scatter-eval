#include "gather/simd_variants/avx512/agg_avx512_64BitVariants.h"
#include "common.cpp"

constexpr bool multi_threaded = false;
constexpr bool avx512 = true;

using ResultT = uint64_t;

// 64 bits? else 32 bit integers
constexpr bool bits64 = std::is_same<ResultT, uint64_t>::value;

int main(int argc, const char** argv) {
    if (argc < 2) {
        cerr << "Data Size as input expected (as log_2)!" << endl;
        return NO_DATA_SIZE_GIVEN;
    }

    int data_size_log2 = atoi(argv[1]);

	const vector<aggregator_t<ResultT>> aggregators	{
		{ aggregate_scalar,					"scalar",	false },
		{ aggregate_linear_avx512_64,			"linear",	false },
		{ aggregate_blockstrided_gather_avx512_64,	"gather",	true },
		{ aggregate_fullstrided_gather_avx512_64,	"gather",	true }
	};
	return main_single_threaded<ResultT>(
		aggregators,
		data_size_log2,	// log2 of number of integers
		multi_threaded,
		avx512,
		bits64
	);
}
