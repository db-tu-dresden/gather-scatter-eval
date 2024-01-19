#ifndef AGG_AVX512_32BITVARIANTS_H
#define AGG_AVX512_32BITVARIANTS_H

#include <immintrin.h>
#include <cstring>

/**
 * @brief scalar variant
 * 
 * @param array 
 * @param number 
 * @return uint64_t 
 */

uint64_t aggregate_scalar(const uint32_t* array, uint64_t number, const uint32_t stride=0) {
        uint64_t res = 0;
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

uint64_t aggregate_linear_avx512_32(const uint32_t* array, uint64_t number, const uint32_t stride=0) {
  __m512i tmp, data;
  uint64_t r = 0;
  
  tmp = _mm512_setzero_si512();
  for (int i = 0; i < number - 16 + 1; i += 16) {
    data = _mm512_load_epi32(reinterpret_cast<const __m512i *> (&array[i]));
    tmp = _mm512_add_epi32(data, tmp);
  }

  return _mm512_reduce_add_epi32(tmp );
}

/**
 * @brief avx512 block strided access variant using gather instruction
 * 
 * @param array 
 * @param number 
 * @param stride 
 * @return uint64_t 
 */

uint64_t aggregate_blockstrided_gather_avx512_32(const uint32_t* array, uint64_t number, const uint32_t stride) {
  __m512i tmp, data;

  tmp = _mm512_setzero_si512();

  const __m512i gatherindex = _mm512_set_epi32(15 * stride, 14 * stride, 13 * stride, 12 * stride, 11 * stride, 10 * stride, 9 * stride, 8 * stride, 7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);

  for (int j = 0; j < number; j += 16 * stride) {
    for (int i = 0; i < stride; i++) {
      data = _mm512_i32gather_epi32(gatherindex, reinterpret_cast<void const *> (&array[j + i]), 4);
      tmp = _mm512_add_epi32(data, tmp);
    }
  }

  return _mm512_reduce_add_epi32 (tmp);
}


/**
 * @brief avx512 block strided access variant using set instruction
 *  
 * @param array 
 * @param number 
 * @param stride 
 * @return uint64_t 
 */
uint64_t aggregate_blockstrided_set_avx512_32(const uint32_t* array, uint64_t number, const uint32_t stride) {
  __m512i tmp, data;

  tmp = _mm512_setzero_si512();

  for (int j = 0; j < number; j += 16 * stride) {
    for (int i = 0; i < stride; i++) {
			data = _mm512_set_epi32(array[j+i+15*stride],array[j+i+14*stride],array[j+i+13*stride],array[j+i+12*stride],array[j+i+11*stride],array[j+i+10*stride],array[j+i+9*stride],array[j+i+8*stride],array[j+i+7*stride],array[j+i+6*stride],array[j+i+5*stride],array[j+i+4*stride],array[j+i+3*stride],array[j+i+2*stride],array[j+i+stride],array[j+i]);
      tmp = _mm512_add_epi64(data, tmp);
    }
  }

  return _mm512_reduce_add_epi32 (tmp);
}




#endif /* AGG_AVX512_32BITVARIANTS_H */
