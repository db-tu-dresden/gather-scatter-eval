#ifndef PARAMETERS_H
#define PARAMETERS_H

#ifdef ITERATIONS
#error ITERATIONS already defined!
#else
#define ITERATIONS 10
#endif

#ifdef MAX_CORES
#error MAX_CORES already defined
#else
// has to be divisible by 2
#define MAX_CORES 64
#endif

#endif // include guard PARAMETERS_H
