cmake_minimum_required(VERSION 3.5)
project(tensorrt_rfbnet)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS  "-lpthread")
set(CMAKE_CXX_FLAGS  "-O3")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -fPIC")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -std=c++11 -fPIC -fpermissive")
endif()
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)


set(OpenCV_DIR /home/tao/OpenCV_CUDA/opencv-3.4.6/build_cuda)
find_package( OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
find_package( CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

#include_directories(/home/tao/object_tracker/person_track/tensorrt_cuda/include)

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
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp )


#AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable(
    ${PROJECT_NAME}
    ${HEADER_FILES}
    ${SOURCE_FILES}
)


target_link_libraries( 
    ${PROJECT_NAME}
	${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}
	/usr/lib/x86_64-linux-gnu/libnvcaffe_parser.so
	/usr/lib/x86_64-linux-gnu/libnvinfer.so
	/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
	/usr/lib/x86_64-linux-gnu/libnvparsers.so
	/usr/lib/x86_64-linux-gnu/libnvonnxparser_runtime.so
    	/usr/lib/x86_64-linux-gnu/libnvonnxparser.so
    	${CMAKE_CURRENT_SOURCE_DIR}/cuda_lib/build/libmatrix.so

)



