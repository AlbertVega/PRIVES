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

# Estimation module (HR / FR via IIR + FFT)
# FFTW3 (spectral analysis, leverages ARM NEON on RPi 5)
pkg_check_modules(FFTW3 REQUIRED fftw3)

# Iir1 (real-time IIR filtering library by Bernd Porr)
find_library(IIR1_LIB NAMES iir
    PATHS /usr/lib /usr/local/lib
    DOC "Iir1 shared library (libiir.so)")
find_path(IIR1_INCLUDE_DIR NAMES Iir.h
    PATHS /usr/include /usr/local/include /usr/include/iir /usr/local/include/iir
    DOC "Iir1 header directory")

if(NOT IIR1_LIB OR NOT IIR1_INCLUDE_DIR)
    message(FATAL_ERROR
        "Iir1 not found. Install via: sudo apt install libiir-dev  or build "
        "from https://github.com/berndporr/iir1")
endif()

# Test executable for estimation (HR + FR processing)
add_executable(test_estimation
    tests/test_estimation.cpp
    estimation/estimation.cpp
)
target_include_directories(test_estimation PRIVATE
    ${FFTW3_INCLUDE_DIRS}
    ${IIR1_INCLUDE_DIR}
)
target_link_libraries(test_estimation PRIVATE
    ${OpenCV_LIBS}
    ${FFTW3_LIBRARIES}
    ${IIR1_LIB}
    pthread
)
