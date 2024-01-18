#ifndef ALLOCATE_CPP
#define ALLOCATE_CPP

#include <numa.h>

template <class ResultT>
ResultT* allocate(
	uint64_t number_of_values,
	uint64_t numa_node = 0
) {
	uint64_t number_of_bytes = number_of_values * sizeof(ResultT);
	ResultT* result = (ResultT*) numa_alloc_onnode(number_of_bytes, numa_node);
	if (!result) {
		cerr
			<< "!!! Failed to allocate !!! "
			<< " had requested " << number_of_bytes
			<< " Bytes on node " << numa_node
		<< endl;
	}
	return result;
}


#endif // include guard ALLOCATE_CPP
