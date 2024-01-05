#ifndef MEASURES_H
#define MEASURES_H

#include <map>

/** meansurement of a benchmark runthrough:
 * result of the measured aggregation function for correctness checking,
 * duration in ns, throughput in GB/s, mis is million values per second.
 */
struct measures {
	uint64_t result;
	double duration;
	double throughput;
	double mis;
};

/** each thread gets to write in its own data result struct
 */
typedef std::map<uint64_t, struct measures> multithreaded_measures;

#endif // include guard MEASURES_H
