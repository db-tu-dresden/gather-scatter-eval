#ifndef AGG_AVX_64BITVARIANTS_H
#define AGG_AVX_64BITVARIANTS_H

#include <immintrin.h>
#include <cstring>
#include <cstdint>

#include "gather/aggregate_scalar.cpp"

/**
 * @brief linear load avx512 variant
 *
 * @param array
 * @param number
 * @return int32_t
 */

uint64_t aggregate_linear_avx256(const uint64_t* array, uint64_t number, const uint32_t stride=0) {
  __m256i tmp, data;
  uint64_t r = 0;

  tmp = _mm256_setzero_si256();
  for (int i = 0; i < number - 4 + 1; i += 4) {
    data = _mm256_load_si256(reinterpret_cast<const __m256i *> (&array[i]));
    tmp  = _mm256_add_epi64(data, tmp);
  }

  uint64_t res = (
    _mm256_extract_epi64(tmp, 0) +
    _mm256_extract_epi64(tmp, 1) +
    _mm256_extract_epi64(tmp, 2) +
    _mm256_extract_epi64(tmp, 3)
  );

  return res;
}

/**
 * @brief avx256 strided access variant  using gather instruction
 *
 * @param array
 * @param number
 * @param stride
 * @return uint64_t
 */

uint64_t aggregate_strided_gather_avx256(const uint64_t* array, uint64_t number, const uint32_t stride) {
  __m256i tmp, data;

  tmp = _mm256_setzero_si256();

  const __m128i gatherindex = _mm_set_epi32(3 * stride, 2 * stride, stride, 0);

  for (int j = 0; j < number; j += 4 * stride) {
    for (int i = 0; i < stride; i++) {
      data = _mm256_i32gather_epi64(reinterpret_cast<const long long int *> (&array[j + i]), gatherindex, 8);
      tmp  = _mm256_add_epi64(data, tmp);
    }
  }

  uint64_t res = (
    _mm256_extract_epi64(tmp, 0) +
    _mm256_extract_epi64(tmp, 1) +
    _mm256_extract_epi64(tmp, 2) +
    _mm256_extract_epi64(tmp, 3)
  );

  return res;
}

/**
 * @brief avx/avx2 strided access variant using set instruction
 *
 * @param array
 * @param number
 * @param stride
 * @return uint64_t
 */
uint64_t aggregate_strided_set_avx256(const uint64_t* array, uint64_t number, const uint32_t stride) {
  __m256i tmp, data;

  tmp = _mm256_setzero_si256();

  for (int j = 0; j < number; j += 4 * stride) {
    for (int i = 0; i < stride; i++) {
      data = _mm256_set_epi64x(array[j+i+3*stride],array[j+i+2*stride],array[j+i+stride],array[j+i]);
      tmp  = _mm256_add_epi64(data, tmp);
    }
  }

  uint64_t res = (
    _mm256_extract_epi64(tmp, 0) +
    _mm256_extract_epi64(tmp, 1) +
    _mm256_extract_epi64(tmp, 2) +
    _mm256_extract_epi64(tmp, 3)
  );

  return res;
}


#endif /* AGG_AVX_64BITVARIANTS_H */
