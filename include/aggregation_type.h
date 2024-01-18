#ifndef AGGREGATION_TYPE_H
#define AGGREGATION_TYPE_H

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
using benchmark_function = aggregation_function_t<ResultT>;


#endif // include guard AGGREGATION_TYPE_H
