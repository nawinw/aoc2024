add_executable(part1 part1.cu)
set_target_properties(part1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(part1 PUBLIC "../include")

add_executable(part2 part2.cu)
target_include_directories(part2 PUBLIC "../include")
