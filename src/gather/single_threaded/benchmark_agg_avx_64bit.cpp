
#include "common.cpp"
#include "gather/simd_variants/avx/agg_avx_64BitVariants.h"

int main(int argc, const char** argv) {
    if (argc < 2) {
        cerr << "Data Size as input expected (as log_2)!" << endl;
        return NO_DATA_SIZE_GIVEN;
    }

    int data_size_log2 = atoi(argv[1]);

	const vector<aggregator_t<uint64_t>> aggregators	{
		{ aggregate_scalar,					"scalar",	false },
		{ aggregate_linear_avx256,			"linear",	false },
		{ aggregate_strided_gather_avx256,	"gather",	true },
		{ aggregate_strided_set_avx512,		"seti",		true },
	};
	return main_single_threaded<uint64_t>(
		aggregators,
		data_size_log2,
		false,
		false,
		true
	);
}
