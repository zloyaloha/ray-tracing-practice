set(SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(COMMON_DIR ../common)

add_executable(main_gpu ../main.cu src/camera.cu)
target_include_directories(main_gpu PUBLIC ${INCLUDE_DIR} ${COMMON_DIR})

target_compile_options(main_gpu
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast-math>
)
