cmake_minimum_required(VERSION 3.10)
project(test)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -ggdb -Wall -w -fPIC -fno-use-cxa-atexit -fexceptions")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

include_directories(${PROJECT_SOURCE_DIR}/src) 
include_directories(${PROJECT_SOURCE_DIR}/external/tflite/include) 

link_directories(${PROJECT_SOURCE_DIR}/external/tflite/lib64) 
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp src/tflite.cpp)
target_link_libraries(${PROJECT_NAME} tensorflowlite ${OpenCV_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "test")
install(TARGETS ${PROJECT_NAME} DESTINATION ${LIBRARY_OUTPUT_PATH}) 

