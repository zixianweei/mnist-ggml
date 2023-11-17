cmake_minimum_required(VERSION 3.16)

project(mnist-ggml VERSION 0.0.1)

enable_language(C)
enable_language(CXX)

###############################################################################
# GGML

add_subdirectory(${CMAKE_SOURCE_DIR}/3dparty/ggml)

###############################################################################
# STB Image

include_directories(${CMAKE_SOURCE_DIR}/3dparty/stb_image)

###############################################################################
# macOS snprintf deprecated warning

add_compile_options("-Wno-deprecated-declarations")

###############################################################################
# ASan

add_compile_options("-fsanitize=address")
add_compile_options("-fno-omit-frame-pointer")
add_link_options("-fsanitize=address")

###############################################################################
# mnist-ggml

set(MNIST_GGML_TARGET_NAME "mnist-ggml")
set(MNIST_GGML_TARGET_SOURCE ${CMAKE_SOURCE_DIR}/source/main.cpp)

add_executable(${MNIST_GGML_TARGET_NAME} ${MNIST_GGML_TARGET_SOURCE})
set_property(TARGET ${MNIST_GGML_TARGET_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${MNIST_GGML_TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(${MNIST_GGML_TARGET_NAME} PUBLIC ggml)
