cmake_minimum_required(VERSION 3.16)
project(PRIVES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Search for installed OpenCV
find_package(OpenCV REQUIRED)

# Includes and dependencies
include_directories(${OpenCV_INCLUDE_DIRS})

# Define the test executable with all its sources
add_executable(test_anonymizer 
    tests/test_anonymizer.cpp 
    anonymizer/Anonymizer.cpp
)

# Link with OpenCV
target_link_libraries(test_anonymizer PRIVATE ${OpenCV_LIBS})

# Test executable for PosExtractor
add_executable(test_pos
    tests/test_pos.cpp
    extraction/PosExtractor.cpp
)
target_link_libraries(test_pos PRIVATE ${OpenCV_LIBS})


# Face Detection module (SCRFD + Hailo-8 NPU + GStreamer capture)

# GStreamer (required for the video capture pipeline)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-app-1.0)

# HailoRT (NPU acceleration library)
find_library(HAILORT_LIB NAMES hailort
    PATHS /usr/lib /usr/local/lib
    DOC "HailoRT shared library (libhailort.so)")
find_path(HAILORT_INCLUDE_DIR NAMES hailo/hailort.hpp
    PATHS /usr/include /usr/local/include
    DOC "HailoRT header directory")

if(NOT HAILORT_LIB OR NOT HAILORT_INCLUDE_DIR)
    message(FATAL_ERROR
        "HailoRT not found. Install the hailort package or set "
        "HAILORT_LIB and HAILORT_INCLUDE_DIR manually.")
endif()

# Test executable for FaceDetectorSCRFD (capture + detection + MJPEG stream)
add_executable(test_capture
    tests/test_capture.cpp
    detection/FaceDetectorSCRFD.cpp
)
target_include_directories(test_capture PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${HAILORT_INCLUDE_DIR}
)
target_link_libraries(test_capture PRIVATE
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${HAILORT_LIB}
    pthread
)
