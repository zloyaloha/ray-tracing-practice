cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(pgrp_labs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler=-std=c++20 --expt-relaxed-constexpr") # Явное указание флага компилятору хоста


set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(cpu)
add_subdirectory(gpu)