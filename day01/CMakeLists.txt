add_executable(day011 part1.cu)
set_target_properties(day011 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(day011 PUBLIC "../include")

add_executable(day012 part2.cu)
set_target_properties(day012 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(day012 PUBLIC "../include")
