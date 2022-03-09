# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/nifty_reg_source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/nifty_reg_build

# Include any dependencies generated for this target.
include reg-lib/CMakeFiles/_reg_femTransformation.dir/depend.make

# Include the progress variables for this target.
include reg-lib/CMakeFiles/_reg_femTransformation.dir/progress.make

# Include the compile flags for this target's objects.
include reg-lib/CMakeFiles/_reg_femTransformation.dir/flags.make

reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o: reg-lib/CMakeFiles/_reg_femTransformation.dir/flags.make
reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o: /workspace/nifty_reg_source/reg-lib/_reg_femTransformation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/nifty_reg_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o"
	cd /workspace/nifty_reg_build/reg-lib && /usr/bin/cc  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o -c /workspace/nifty_reg_source/reg-lib/_reg_femTransformation.cpp

reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.i"
	cd /workspace/nifty_reg_build/reg-lib && /usr/bin/cc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/nifty_reg_source/reg-lib/_reg_femTransformation.cpp > CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.i

reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.s"
	cd /workspace/nifty_reg_build/reg-lib && /usr/bin/cc $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/nifty_reg_source/reg-lib/_reg_femTransformation.cpp -o CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.s

reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o.requires:

.PHONY : reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o.requires

reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o.provides: reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o.requires
	$(MAKE) -f reg-lib/CMakeFiles/_reg_femTransformation.dir/build.make reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o.provides.build
.PHONY : reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o.provides

reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o.provides.build: reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o


# Object files for target _reg_femTransformation
_reg_femTransformation_OBJECTS = \
"CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o"

# External object files for target _reg_femTransformation
_reg_femTransformation_EXTERNAL_OBJECTS =

reg-lib/lib_reg_femTransformation.a: reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o
reg-lib/lib_reg_femTransformation.a: reg-lib/CMakeFiles/_reg_femTransformation.dir/build.make
reg-lib/lib_reg_femTransformation.a: reg-lib/CMakeFiles/_reg_femTransformation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/nifty_reg_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library lib_reg_femTransformation.a"
	cd /workspace/nifty_reg_build/reg-lib && $(CMAKE_COMMAND) -P CMakeFiles/_reg_femTransformation.dir/cmake_clean_target.cmake
	cd /workspace/nifty_reg_build/reg-lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_reg_femTransformation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
reg-lib/CMakeFiles/_reg_femTransformation.dir/build: reg-lib/lib_reg_femTransformation.a

.PHONY : reg-lib/CMakeFiles/_reg_femTransformation.dir/build

reg-lib/CMakeFiles/_reg_femTransformation.dir/requires: reg-lib/CMakeFiles/_reg_femTransformation.dir/_reg_femTransformation.cpp.o.requires

.PHONY : reg-lib/CMakeFiles/_reg_femTransformation.dir/requires

reg-lib/CMakeFiles/_reg_femTransformation.dir/clean:
	cd /workspace/nifty_reg_build/reg-lib && $(CMAKE_COMMAND) -P CMakeFiles/_reg_femTransformation.dir/cmake_clean.cmake
.PHONY : reg-lib/CMakeFiles/_reg_femTransformation.dir/clean

reg-lib/CMakeFiles/_reg_femTransformation.dir/depend:
	cd /workspace/nifty_reg_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/nifty_reg_source /workspace/nifty_reg_source/reg-lib /workspace/nifty_reg_build /workspace/nifty_reg_build/reg-lib /workspace/nifty_reg_build/reg-lib/CMakeFiles/_reg_femTransformation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : reg-lib/CMakeFiles/_reg_femTransformation.dir/depend

