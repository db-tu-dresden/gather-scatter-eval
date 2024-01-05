#ifndef GENERATE_RANDOM_VALUES_CPP
#define GENERATE_RANDOM_VALUES_CPP

/** uses std::mt19937 seeded with std::random_device and the current time
 * to write values from a std::uniform_int_distribution into
 * all number fields of the array
 */
template <typename T>
void generate_random_values(T* array, uint64_t number) {
  static_assert(is_integral<T>::value, "Data type is not integral.");
  std::random_device rd;
  std::mt19937::result_type seed = rd() ^ (
          (std::mt19937::result_type)
          std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch()
          ).count() +
          (std::mt19937::result_type)
          std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch()
          ).count());

  std::mt19937 gen(seed);
  std::uniform_int_distribution<T> distrib(1, 6);

  for (uint64_t j = 0; j < number; ++j) {
    array[j] = distrib(gen);
  }
}


#endif // include guard GENERATE_RANDOM_VALUES_CPP
