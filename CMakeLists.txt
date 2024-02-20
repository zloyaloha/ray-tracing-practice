cmake_minimum_required(VERSION 3.10)
project(vec3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(HITTABLE_DIR ${SOURCE_DIR}/HittableObjects)
set(MATERIALS_DIR ${SOURCE_DIR}/Materials)

include_directories(${SOURCE_DIR})
include_directories(${INCLUDE_DIR})
include_directories(${MATERIALS_DIR})
include_directories(${HITTABLE_DIR})

add_library(hitObj_lib ${HITTABLE_DIR}/hittable_object.cpp ${HITTABLE_DIR}/sphere.cpp ${HITTABLE_DIR}/hittable_list.cpp)
add_library(vec3_lib ${SOURCE_DIR}/vec3.cpp)
add_library(interval_lib ${SOURCE_DIR}/interval.cpp)
add_library(ray_lib ${SOURCE_DIR}/ray.cpp)
add_library(materials_lib ${MATERIALS_DIR}/lambertian.cpp)
add_library(camera_lib ${SOURCE_DIR}/camera.cpp)

add_executable(main.exe main.cpp)

target_link_libraries(hitObj_lib PUBLIC vec3_lib ray_lib interval_lib)

target_link_libraries(main.exe PUBLIC vec3_lib
                                interval_lib
                                ray_lib           
                                hitObj_lib
                                materials_lib
                                camera_lib
                                )