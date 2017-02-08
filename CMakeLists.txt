project (cudatest)

# Lowest supported cmake compatible with cmake 3.5
cmake_minimum_required(VERSION 2.4)

# Suggested by CUDA package
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

find_package(CUDA REQUIRED)

cuda_add_executable (cuda-test cuda_test.cu)