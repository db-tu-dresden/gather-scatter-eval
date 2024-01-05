#ifndef CREATE_THREAD_CPP
#define CREATE_THREAD_CPP

#include "aggregation_type.h"

/** creates a std::thread with the given id, pointers to return thread data by,
 * the passed sync_barrier and the aggregation func to be measured.
 * the thread is pinned to the cpu with tid via pthread_setaffinity_np.
 * returns the created thread.
 */
template< typename Function, class ResultT>
std::thread* create_thread(
	const uint64_t tid,
	ResultT* local_result,
	double* local_duration,
	bool* local_ready,
	std::shared_future< void >* sync_barrier,
	Function&& magic,
	benchmark_function<ResultT> func
) {
    cpu_set_t cpuset;
    CPU_ZERO( &cpuset );
    CPU_SET( tid, &cpuset );
    std::thread* t = new std::thread(
		std::forward< Function >( magic ),
		tid,
		local_result,
		local_duration,
		local_ready,
		sync_barrier,
		func
	);
    int rc = pthread_setaffinity_np( t->native_handle(), sizeof( cpu_set_t ), &cpuset );
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        exit( -10 );
    }
    return t;
}


#endif // include guard CREATE_THREAD_CPP
