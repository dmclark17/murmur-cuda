cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(murmur-cuda LANGUAGES CXX CUDA)

add_library(murmur-cuda STATIC
  src/murmur3.cu
  src/murmur3.h
  )

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(murmur-cuda PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( murmur-cuda
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

add_executable(main main.cu)

target_link_libraries(main murmur-cuda)
target_include_directories(main PUBLIC src)

set_target_properties(main PROPERTIES CXX_STANDARD 11)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
