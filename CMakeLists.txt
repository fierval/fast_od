cmake_minimum_required(VERSION 3.8)
project(tf_detector_example LANGUAGES CXX CUDA)

cmake_policy(SET CMP0074 OLD)

set(CMAKE_CXX_STANDARD 11)

# CUDA for cudacodec ops
find_package(CUDA 9.0 REQUIRED)

set(SOURCE_FILES 
    main.cpp
    utils.cpp 
    utils.h 
    dynlink_nvcuvid.cpp 
    infer_with_trt.cpp
    inference_base.cpp 
    inference_tf.cpp
    inference_trt.cpp
    channel_first.cu
    )

# Tensorflow directories and libraries
set(TENSORFLOW_LIBS libtensorflow_cc.so libtensorflow_framework.so)
set(MYHOME $ENV{HOME})

message("-- Home set to: " ${MYHOME})
link_directories("/usr/local/tensorflow/lib")

add_executable(tf_detector_example ${SOURCE_FILES})
set_target_properties(tf_detector_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# OpenCV libs
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# ==================== PATHS TO SPECIFY! ==================== #

# TensorFlow headers
include_directories("/usr/local/tensorflow/include/tensorflow/")
include_directories("/usr/local/tensorflow/include/third-party/")
include_directories("/usr/local/tensorflow/include/")

# IMPORTANT: Protobuf includes. Depends on the anaconda path
# This is Azure DLVM (not sure if DSVM is the same)
include_directories("/data/anaconda/envs/py36/lib/python3.6/site-packages/tensorflow/include/")
# This is a standard install of Anaconda with p36 environment
include_directories("${MYHOME}/anaconda3/envs/py36/lib/python3.6/site-packages/tensorflow/include/")

target_link_libraries(tf_detector_example
    ${CUDA_LIBRARIES}
    cuda
    cublas
    nvinfer
    nvToolsExt
    nvparsers 
    nvinfer_plugin 
    nvonnxparser 
    ${CMAKE_DL_LIBS}
    ${OpenCV_LIBS} 
    ${TENSORFLOW_LIBS})
