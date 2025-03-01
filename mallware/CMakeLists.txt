cmake_minimum_required(VERSION 3.15)
project(NMCardMini_Inference)

set(CMAKE_CXX_STANDARD 17)

# Задайте пути к вашим заголовочным файлам и библиотекам NMDL
set(INC_DIR "/opt/nmdl/include")
set(LIB_DIR "/opt/nmdl/lib")

add_executable(${PROJECT_NAME} inference.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${INC_DIR})
target_link_directories(${PROJECT_NAME} PUBLIC ${LIB_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC nmdl nmdl_compiler nmdl_image_converter)

