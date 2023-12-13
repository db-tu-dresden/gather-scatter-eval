#ifndef MAKE_LABEL_CPP
#define MAKE_LABEL_CPP

std::string make_label(
	uint64_t data_size_log2,
	bool multi_threaded,
	bool avx512,
	bool bits64,
	std::string sep = "_",
	bool include_data_size = true
) {
	std::string result = "";
	if (include_data_size)
		result += std::to_string(data_size_log2) + sep;
	result += (multi_threaded ? "multi_threaded" : "single_threaded") + sep;
	result += (avx512         ? "avx512" : "avx256") + sep;
	result += (bits64         ? "64bit" : "32bit");
	return result;
}


#endif // include guare MAKE_LABEL_CPP
