
project(cuda_opencv_v1)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON) 
set(OpenCV_DIR /home/tao/OpenCV_CUDA/opencv-3.4.6/build_cuda)
find_package(OpenCV REQUIRED)
find_package( CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};--disable-warnings;
        -O3
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_50,code=compute_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_62,code=sm_62
        )
file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)

CUDA_ADD_LIBRARY(matrix  SHARED ${SOURCE_FILES} ${HEADER_FILES})

