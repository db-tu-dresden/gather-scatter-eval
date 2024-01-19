#ifndef AGG_AVX512_64BITVARIANTS_H
#define AGG_AVX512_64BITVARIANTS_H

#include <immintrin.h>
#include <cstring>
#include <cstdint>
#include <math.h>

#include "gather/aggregate_scalar.cpp"

/**
 * @brief linear load avx512 variant
 *
 * @param array
 * @param number
 * @return int64_t
 */

uint64_t aggregate_linear_avx512_64(const uint64_t* array, uint64_t number, const uint32_t stride=0) {
  __m512i tmp, data;
  uint64_t r = 0;

  tmp = _mm512_setzero_si512();
  for (int i = 0; i < number - 8 + 1; i += 8) {
    data = _mm512_load_epi64(reinterpret_cast<const __m512i *> (&array[i]));
    tmp = _mm512_add_epi64(data, tmp);
  }

  return _mm512_reduce_add_epi64(tmp );
}


/**
 * @brief avx512 strided access variant using gather instruction
 *
 * @param array
 * @param number
 * @param stride
 * @return uint64_t
 */

uint64_t aggregate_blockstrided_gather_avx512_64(const uint64_t* array, uint64_t number, const uint32_t stride) {
  __m512i tmp, data;

  tmp = _mm512_setzero_si512();

  const __m256i gatherindex = _mm256_set_epi32(7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);

  for (int j = 0; j < number; j += 8 * stride) {
    for (int i = 0; i < stride; i++) {
      data = _mm512_i32gather_epi64(gatherindex, reinterpret_cast<void const *> (&array[j + i]), 8);
      tmp = _mm512_add_epi64(data, tmp);
    }
  }
  return _mm512_reduce_add_epi64 (tmp);
}

uint64_t aggregate_fullstrided_gather_avx512_64(const uint64_t* array, uint64_t number, const uint32_t stride) {
  __m512i tmp, data;

  tmp = _mm512_setzero_si512();

  const __m256i gatherindex = _mm256_set_epi32(7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);

  for (int i = 0; i < stride; i++) {
    for (int j = 0; j < number; j += 8 * stride) {
      data = _mm512_i32gather_epi64(gatherindex, reinterpret_cast<void const *> (&array[j + i]), 8);
      tmp = _mm512_add_epi64(data, tmp);
    }
  }
  return _mm512_reduce_add_epi64 (tmp);
}




/**
 * @brief avx512 strided access variant using set instruction
 *
 * @param array
 * @param number
 * @param stride
 * @return uint64_t
 */
uint64_t aggregate_blockstrided_set_avx512_64(const uint64_t* array, uint64_t number, const uint32_t stride) {
  __m512i tmp, data;

  tmp = _mm512_setzero_si512();

  for (int j = 0; j < number; j += 8 * stride) {
    for (int i = 0; i < stride; i++) {
			data = _mm512_set_epi64(array[j+i+7*stride],array[j+i+6*stride],array[j+i+5*stride],array[j+i+4*stride],array[j+i+3*stride],array[j+i+2*stride],array[j+i+stride],array[j+i]);
      tmp = _mm512_add_epi64(data, tmp);
    }
  }

  return _mm512_reduce_add_epi64 (tmp);
}

#endif /* AGG_AVX512_64BITVARIANTS_H */
