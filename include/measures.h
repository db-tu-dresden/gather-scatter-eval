#ifndef MEASURES_H
#define MEASURES_H

#include <map>

struct measures {
	uint64_t result;
	double duration;
	double throughput;
	double mis;
};

typedef std::map<uint64_t, struct measures> multithreaded_measures;

#endif // include guard MEASURES_H
