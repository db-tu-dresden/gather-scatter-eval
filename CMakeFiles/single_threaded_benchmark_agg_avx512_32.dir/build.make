# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/habich/github2/gather-scatter-eval

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/habich/github2/gather-scatter-eval

# Include any dependencies generated for this target.
include CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/flags.make

CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.o: CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/flags.make
CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.o: src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/habich/github2/gather-scatter-eval/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.o -c /home/habich/github2/gather-scatter-eval/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp

CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/habich/github2/gather-scatter-eval/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp > CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.i

CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/habich/github2/gather-scatter-eval/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp -o CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.s

# Object files for target single_threaded_benchmark_agg_avx512_32
single_threaded_benchmark_agg_avx512_32_OBJECTS = \
"CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.o"

# External object files for target single_threaded_benchmark_agg_avx512_32
single_threaded_benchmark_agg_avx512_32_EXTERNAL_OBJECTS =

bin/single_threaded_benchmark_agg_avx512_32: CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/src/gather/single_threaded/benchmark_agg_avx512_32bit.cpp.o
bin/single_threaded_benchmark_agg_avx512_32: CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/build.make
bin/single_threaded_benchmark_agg_avx512_32: CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/habich/github2/gather-scatter-eval/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/single_threaded_benchmark_agg_avx512_32"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/build: bin/single_threaded_benchmark_agg_avx512_32

.PHONY : CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/build

CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/cmake_clean.cmake
.PHONY : CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/clean

CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/depend:
	cd /home/habich/github2/gather-scatter-eval && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/habich/github2/gather-scatter-eval /home/habich/github2/gather-scatter-eval /home/habich/github2/gather-scatter-eval /home/habich/github2/gather-scatter-eval /home/habich/github2/gather-scatter-eval/CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/single_threaded_benchmark_agg_avx512_32.dir/depend

