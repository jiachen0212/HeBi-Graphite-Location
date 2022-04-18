cmake_minimum_required(VERSION 3.11-3.18)

project(
        example
        VERSION 1.0.0
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/3rdparty/opencv_linux/lib/cmake/opencv4)
    set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/3rdparty/eigen/Eigen3/share/eigen3/cmake)
    set(smorevision_DIR ${PROJECT_SOURCE_DIR}/3rdparty/smorevision/smorevision_linux_x86_64_Release/lib/cmake)
elseif (CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/3rdparty/opencv_win/opencv/cmake)
    set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/3rdparty/eigen/Eigen3/share/eigen3/cmake)
    set(smorevision_DIR ${PROJECT_SOURCE_DIR}/3rdparty/smorevision/smorevision_win_x86_64_Release/lib/cmake)
else()
    set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/3rdparty/opencv_linux/lib/cmake/opencv4)
    set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/3rdparty/eigen/Eigen3/share/eigen3/cmake)
    set(smorevision_DIR ${PROJECT_SOURCE_DIR}/3rdparty/smorevision/smorevision_linux_x86_64_Release/lib/cmake)
endif ()

find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
find_package(Eigen3 REQUIRED PATHS ${Eigen3_DIR})
find_package(smorevision CONFIG REQUIRED)

add_subdirectory(3rdparty/json/json-3.9.1)

add_executable(example1 example1.cpp location.cpp location.h)
target_include_directories(example1
        PRIVATE
        ${smorevision_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(example1
        PRIVATE
        ${smorevision_LIBRARIES}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        )


add_executable(example2 example2.cpp location.cpp location.h)
target_include_directories(example2
        PRIVATE
        ${smorevision_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(example2
        PRIVATE
        ${smorevision_LIBRARIES}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        )


add_executable(example3 example3.cpp location.cpp location.h)
target_include_directories(example3
        PRIVATE
        ${smorevision_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(example3
        PRIVATE
        ${smorevision_LIBRARIES}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        )


add_executable(example4 example4.cpp location.cpp location.h)
target_include_directories(example4
        PRIVATE
        ${smorevision_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(example4
        PRIVATE
        ${smorevision_LIBRARIES}
        ${OpenCV_LIBS}
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        )