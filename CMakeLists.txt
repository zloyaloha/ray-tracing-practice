cmake_minimum_required(VERSION 3.10)
project(vec3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(HITTABLE_DIR ${SOURCE_DIR}/HittableObjects)
set(MATERIALS_DIR ${SOURCE_DIR}/Materials)

add_library(interval_lib ${SOURCE_DIR}/interval.cpp)
add_library(ray_lib ${SOURCE_DIR}/ray.cpp)

add_library(hitObj_lib
  ${HITTABLE_DIR}/hittable_object.cpp
  ${HITTABLE_DIR}/sphere.cpp
  ${HITTABLE_DIR}/plane.cpp
  ${HITTABLE_DIR}/hittable_list.cpp
)

add_library(materials_lib
  ${MATERIALS_DIR}/lambertian.cpp
  ${MATERIALS_DIR}/metal.cpp
  ${MATERIALS_DIR}/dielectric.cpp
)

add_library(camera_lib ${SOURCE_DIR}/camera.cpp)

# Правильная расстановка зависимостей
target_include_directories(ray_lib PUBLIC ${INCLUDE_DIR})
target_include_directories(interval_lib PUBLIC ${INCLUDE_DIR})
target_include_directories(hitObj_lib PUBLIC ${INCLUDE_DIR})
target_include_directories(materials_lib PUBLIC ${INCLUDE_DIR})
target_include_directories(camera_lib PUBLIC ${INCLUDE_DIR})

target_link_libraries(ray_lib PUBLIC  interval_lib)
target_link_libraries(hitObj_lib PUBLIC  ray_lib interval_lib)
target_link_libraries(materials_lib PUBLIC )
target_link_libraries(camera_lib PUBLIC  ray_lib)

# Исполняемый файл
add_executable(main main.cpp)
target_link_libraries(main PRIVATE interval_lib ray_lib hitObj_lib materials_lib camera_lib)