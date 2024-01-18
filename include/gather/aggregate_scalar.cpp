#ifndef AGGREGATE_SCALAR_CPP
#define AGGREGATE_SCALAR_CPP

#include <cstdint>

inline
uint64_t aggregate_scalar(const uint64_t* array, uint64_t number, const uint32_t stride=0) {
	uint64_t res = 0;
	for (uint64_t i = 0; i < number; i++)
		res += array[i];
	return res;
}

inline
uint64_t aggregate_scalar(const uint32_t* array, uint64_t number, const uint32_t stride=0) {
	uint32_t res = 0;
	for (uint64_t i = 0; i < number; i++)
		res += array[i];
	return res;
}

#endif // include guard AGGREGATE_SCALAR_CPP
