#ifndef AGG_AVX_32BITVARIANTS_H
#define AGG_AVX_32BITVARIANTS_H

#include <immintrin.h>
#include <cstring>

/**
 * @brief scalar variant
 * 
 * @param array 
 * @param number 
 * @return uint64_t 
 */

uint32_t aggregate_scalar(const uint32_t* array, uint64_t number, const uint32_t stride=0) {
        uint32_t res = 0;
        for (uint64_t i = 0; i < number; i++)
            res += array[i];
     return res; 
}

/**
 * @brief linear load avx512 variant
 * 
 * @param array 
 * @param number 
 * @return int32_t 
 */

uint64_t aggregate_stream_linear_avx256_32(const uint32_t* array, uint64_t number, const uint32_t stride=0) {
  __m256i tmp, data;
  uint64_t r = 0;
  
  tmp = _mm256_setzero_si256();
  for (int i = 0; i < number - 8 + 1; i += 8) {
    data = _mm256_stream_load_si256(reinterpret_cast<const __m256i *> (&array[i]));

    tmp = _mm256_add_epi32(data, tmp);
  }

   uint64_t res = 0;
  for (int i= 0; i<8; i++)
    res += _mm256_extract_epi32(tmp,i);

  return res;
}

/**
 * @brief linear load avx512 variant
 * 
 * @param array 
 * @param number 
 * @return int32_t 
 */

uint64_t aggregate_linear_avx256_32(const uint32_t* array, uint64_t number, const uint32_t stride=0) {
  __m256i tmp, data;
  uint64_t r = 0;
  
  tmp = _mm256_setzero_si256();
  for (int i = 0; i < number - 8 + 1; i += 8) {
    data = _mm256_load_si256(reinterpret_cast<const __m256i *> (&array[i]));
    tmp = _mm256_add_epi32(data, tmp);
  }

   uint64_t res = 0;
  for (int i= 0; i<8; i++)
    res += _mm256_extract_epi32(tmp,i);

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

uint64_t aggregate_blockstrided_gather_avx256_32(const uint32_t* array, uint64_t number, const uint32_t stride) {
  __m256i tmp, data;

  tmp = _mm256_setzero_si256();

  const __m256i gatherindex = _mm256_set_epi32(7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);

  for (int j = 0; j < number; j += 8 * stride) {
    for (int i = 0; i < stride; i++) {
      data = _mm256_i32gather_epi32(reinterpret_cast<int const *> (&array[j + i]), gatherindex, 4);
      tmp = _mm256_add_epi32(data, tmp);
    }
  }

  uint64_t res = 0;
  for (int i= 0; i<8; i++)
    res += _mm256_extract_epi32(tmp,i);

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
uint64_t aggregate_blockstrided_set_avx256_32(const uint32_t* array, uint64_t number, const uint32_t stride) {
  __m256i tmp, data;

  tmp = _mm256_setzero_si256();

  for (int j = 0; j < number; j += 8 * stride) {
    for (int i = 0; i < stride; i++) {
			data = _mm256_set_epi32(array[j+i+7*stride],array[j+i+6*stride],array[j+i+5*stride],array[j+i+4*stride],array[j+i+3*stride],array[j+i+2*stride],array[j+i+stride],array[j+i]);
      tmp = _mm256_add_epi32(data, tmp);
    }
  }

  uint64_t res = 0;
  for (int i= 0; i<8; i++)
    res += _mm256_extract_epi32(tmp,i);

  return res;
}


#endif /* AGG_AVX_32BITVARIANTS_H */
