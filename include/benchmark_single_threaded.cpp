#ifndef BENCHMARK_SINGLE_THREADED_CPP
#define BENCHMARK_SINGLE_THREADED_CPP

#include "measures.h"
#include "parameters.h"

/** runs a benchmark on the passed function over the given values
 * and stores duration, throughput, result and mis in the struct measures.
 * mis is million values per second.
 * you pass the number of gigabytes that values contains for some reason.
 * some functions take a stride argument, if yours doesn’t, a 0 should work fine.
 * flushes caches and TLB between every function execution, of which there are
 * ITERATIONS many. #defined in parameters.h
 * returns true if the result of the function matches the passed correct result,
 * else false.
 */
template <class ResultT>
bool benchmark(
	measures* res,
	uint64_t correct_result,
	const ResultT* values,
	uint64_t n,
	const uint32_t stride,
	double GB,
	uint64_t (*func)(const ResultT*, uint64_t, const uint32_t)
) {

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



#endif // include guard BENCHMARK_SINGLE_THREADED_CPP
