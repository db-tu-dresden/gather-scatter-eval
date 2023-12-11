# gather-scatter-eval

This contains some testing/benchmarking code for “gather” vector-instructions,
which I (Anton Obersteiner) want to adapt for the XeonMAX processor.

## rough structure
### ./include/gather/simd_variants/avx

#### `agg_avx_($bits:32|64)BitVariants.h`
all these take the following arguments:
```cpp
u$bits* array: the values to aggregate (sum)
u64 number: number of values to aggregate
u32 stride: stride of the aggregation, has default 0 where useless
```

these versions are implemented:
```cpp
step := 256/$bits
bytes_per_number := $bits/8

u$bits aggregate_scalar(...) {
	for i++ { res += array[i]; }
}
u64 aggregate_stream_linear_avx256(...) {
	for i+=step { stream_load(array[i:i+step]); }
}
u64 aggregate_linear_avx256(...) {
	for i+=step { load(array[i:i+step]); }
}
u64 aggregate_strided_gather_avx256(...) {
	gatherindex = set_epi32((step-1) * stride, …, 1*stride, 0*stride);
	for j+=step*stride {
		for i++ < stride {
			i32gather_epi32(array+j+i, gatherindex, bytes_per_number);
		}
	}
}
u64 aggregate_strided_set_avx256(...) {
	for j, i as above {
		set_epi32(
			array[j+i+(step-1)*stride],
			...,
			array[j+i+     1 * stride],
			array[j+i]
		);
	}
}
```

### `./include/gather/avx512`

largely the same (more bits of course) as avx256.
#### `…32BitVariant.h`
`aggregate_strided_gather_avx` has a variant with `_512`
that has fixed stride 512, probably for testing different
non-equidistant offset patterns in `gatherindex`.

syntax differences:
- `gather(gatherindex, array, bytes_per_number)`
- `reduce_add_epi` instead of `+` of `for` adding

#### `…64BitVariant.h`
has a `_64` variant of `aggregate_indexed_gather`
which uses some weird set of prime numbers larger than $step
to choose which values to add. probably to test what non-standard
access patterns do with performance while still gathering all values?

the other one (comment `stride size 512`) is also complicated.

