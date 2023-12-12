#ifndef LOG_MULTITHREADED_RESULTS_CPP
#define LOG_MULTITHREADED_RESULTS_CPP

/* We anticipate the following order: scalar, linear, gather, seti */
void log_multithreaded_results_per_file( std::string basename, const size_t stride_size, std::vector< multithreaded_measures* > results, bool clean ) {
    std::vector< uint64_t > keys;
    /* Extract all keys */
    for ( auto it = results[0]->begin(); it != results[0]->end(); ++it ) {
        keys.push_back( it->first );
    }

    if ( clean ) {
        for ( auto key : keys ) {
            std::string del_file = basename + "_" + std::to_string( key ) + "_cores.dat";
            if ( remove( del_file.c_str() ) == 0 ) {
                std::cout << "Succesfully removed " << del_file << " before the benchmark." << std::endl;
            } else {
                std::cout << "ERROR removing " << del_file << " before the benchmark (maybe file was not present anyway). CHECK RESULTS" << std::endl;
            }
        }
    }

    for ( auto key : keys ) {
        std::ofstream out( basename + "_" + std::to_string( key ) + "_cores.dat", std::ios_base::app );
        out << stride_size << " " << stride_size * 8;
        for ( auto r : results ) {
            out << " " << (*r)[ key ].mis << " " << (*r)[ key ].throughput;
        }
        out << std::endl;
        out.close();
    }
}

void print_multithreaded_results( std::ostream& logfile, std::string ident, multithreaded_measures& results ) {
    for ( auto it = results.begin(); it != results.end(); ++it ) {
        logfile << "[" << ident << "] Core Count: " << it->first << " TPut: " << it->second.throughput << std::endl;
    }
}


#endif
