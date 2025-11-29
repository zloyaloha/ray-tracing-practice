cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(pgrp_labs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(main src/main.cu src/camera.cu)
target_include_directories(main PUBLIC ${INCLUDE_DIR})

target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --ptxas-options=-v
        -lineinfo
        --extra-device-vectorization
    >
)