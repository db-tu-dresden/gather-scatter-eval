#ifndef AGG_AVX512_64BITVARIANTS_H
#define AGG_AVX512_64BITVARIANTS_H

#include <immintrin.h>
#include <cstring>

/**
 * @brief scalar variant
 * 
 * @param array 
 * @param number 
 * @return uint64_t 
 */

uint64_t aggregate_scalar(const uint64_t* array, uint64_t number, const uint32_t stride=0) {
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
 * @return int64_t 
 */

uint64_t aggregate_linear_avx512(const uint64_t* array, uint64_t number, const uint32_t stride=0) {
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

uint64_t aggregate_strided_gather_avx512(const uint64_t* array, uint64_t number, const uint32_t stride) {
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


/**
 * stride size 64
 */

uint64_t aggregate_indexed_gather_avx512_64(const uint64_t* array, uint64_t number,const uint32_t stride) {
  __m512i tmp, data;

  tmp = _mm512_setzero_si512();

  const __m256i gatherindex = _mm256_set_epi32(7 * stride, 6 * stride, 5 * stride, 4 * stride, 3 * stride, 2 * stride, stride, 0);
    

  for (int j = 0; j < number; j += 8 * stride) {
    __m256i positions = _mm256_set_epi32(11, 13, 17, 19, 23, 29, 31, 37);

    for (int i = 0; i < stride; i++) {
      __m256i gatherpos = _mm256_and_si256(positions, _mm256_set_epi32(63, 63, 63, 63, 63, 63, 63, 63));
      /*for (int x=0; x<8; x++) {
        std::cout <<_mm256_extract_epi32(gatherpos,0)<<" ";
      }*/
      //std::cout <<std::endl;
      gatherpos = _mm256_add_epi32(gatherpos, gatherindex);
      data = _mm512_i32gather_epi64(gatherpos, reinterpret_cast<void const *> (&array[j]), 8);
      tmp = _mm512_add_epi64(data, tmp);
      positions = _mm256_add_epi32(positions, _mm256_set_epi32(11, 13, 17, 19, 23, 29, 31, 37));
    }
  }
  return _mm512_reduce_add_epi64 (tmp);
}

/**
 * stride size 512
 */

uint64_t aggregate_indexed_gather_avx512(const uint64_t* array, uint64_t number,const uint64_t stride) {
  __m512i tmp, data;

  tmp = _mm512_setzero_si512();

  const uint64_t stride_size = pow(2,stride);
  const uint32_t f = pow(2,(stride-1));

  //std::cout <<stride_size<<" "<<f<<std::endl;

  const __m256i gatherindex = _mm256_set_epi32(7 * stride_size, 6 * stride_size, 5 * stride_size, 4 * stride_size, 3 * stride_size, 2 * stride_size, stride_size, 0);
  const __m256i fixed1 = _mm256_set_epi32(stride_size-1, stride_size-1, stride_size-1, stride_size-1, stride_size-1, stride_size-1, stride_size-1, stride_size-1);
  const __m256i fixed2 = _mm256_set_epi32(f+1, f+3, f+5, f+9, f+11, f+13, f+15, f+17);
    

  for (int j = 0; j < number; j += 8 * stride_size) {
    __m256i positions = _mm256_set_epi32(f+1, f+3, f+5, f+9, f+11, f+13, f+15, f+17);

    for (int i = 0; i < stride_size; i++) {
      __m256i gatherpos = _mm256_and_si256(positions, fixed1);
      gatherpos = _mm256_add_epi32(gatherpos, gatherindex);
      data = _mm512_i32gather_epi64(gatherpos, reinterpret_cast<void const *> (&array[j]), 8);
      tmp = _mm512_add_epi64(data, tmp);
      positions = _mm256_add_epi32(positions, fixed2);
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
uint64_t aggregate_strided_set_avx512(const uint64_t* array, uint64_t number, const uint32_t stride) {
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
