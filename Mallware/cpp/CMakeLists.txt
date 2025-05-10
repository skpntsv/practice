cmake_minimum_required(VERSION 3.15)

project(mallware)

set(CMAKE_CXX_STANDARD 11)

set(BIN_DIR /opt/nmdl/bin)
set(INC_DIR /opt/nmdl/include)
set(LIB_DIR /opt/nmdl/lib)

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${INC_DIR})

target_link_directories(${PROJECT_NAME} PUBLIC ${LIB_DIR})
			
target_link_libraries(${PROJECT_NAME} PUBLIC nmdl nmdl_compiler nmdl_image_converter)

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${BIN_DIR})

set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_custom_command (
	TARGET ${PROJECT_NAME}
	POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${BIN_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${BIN_DIR}
)
