cmake_minimum_required(VERSION 3.15)
project(ArmorDetection)

include_directories(src/include ${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)

add_executable(armor_detection src/main.cpp)

target_link_libraries(armor_detection ${OpenCV_LIBS})