cmake_minimum_required(VERSION 3.10)
project(vec3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(HITTABLE_DIR ${SOURCE_DIR}/HittableObjects)
set(VECTOR_DIR ${SOURCE_DIR}/Vector)
set(CAMERA_DIR ${SOURCE_DIR}/Camera)

include_directories(${HITTABLE_DIR})
include_directories(${VECTOR_DIR})
include_directories(${CAMERA_DIR})
include_directories(${INCLUDE_DIR})

add_library(vec3_lib ${VECTOR_DIR}/vec3.cpp)
add_library(tracing_lib ${CAMERA_DIR}/camera.cpp ${HITTABLE_DIR}/sphere.cpp ${HITTABLE_DIR}/hittable_list.cpp)

add_executable(main.exe main.cpp)

target_link_libraries(main.exe PRIVATE tracing_lib vec3_lib)