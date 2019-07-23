add_library(murmurlib-cuda STATIC
  murmuda3.cu
  murmuda3.h
  )

target_compile_features(murmurlib-cuda PUBLIC cxx_std_11)
set_target_properties(murmurlib-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
