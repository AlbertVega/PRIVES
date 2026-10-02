cmake_minimum_required(VERSION 3.16)
project(PRIVES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Core dependency (always required)
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# Catch2 v3 (fetched automatically - no manual installation required)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

# ---------------------------------------------------------------------------
# Optional: FFTW3 + Iir1 (required for estimation module)
# ---------------------------------------------------------------------------
find_package(PkgConfig QUIET)

set(HAVE_FFTW3 OFF)
set(HAVE_IIR1  OFF)

if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET fftw3)
    if(FFTW3_FOUND)
        set(HAVE_FFTW3 ON)
    endif()
endif()

find_library(IIR1_LIB NAMES iir
    PATHS /usr/lib /usr/local/lib
    DOC "Iir1 shared library (libiir.so)")
find_path(IIR1_INCLUDE_DIR NAMES Iir.h
    PATHS /usr/include /usr/local/include /usr/include/iir /usr/local/include/iir
    DOC "Iir1 header directory")

if(IIR1_LIB AND IIR1_INCLUDE_DIR)
    set(HAVE_IIR1 ON)
endif()

# ---------------------------------------------------------------------------
# Optional: HailoRT + GStreamer (required for the demo - hardware only)
# ---------------------------------------------------------------------------
set(HAVE_HAILORT OFF)
set(HAVE_GSTREAMER OFF)

find_library(HAILORT_LIB NAMES hailort
    PATHS /usr/lib /usr/local/lib
    DOC "HailoRT shared library (libhailort.so)")
find_path(HAILORT_INCLUDE_DIR NAMES hailo/hailort.hpp
    PATHS /usr/include /usr/local/include
    DOC "HailoRT header directory")

if(HAILORT_LIB AND HAILORT_INCLUDE_DIR)
    set(HAVE_HAILORT ON)
endif()

if(PkgConfig_FOUND)
    pkg_check_modules(GSTREAMER QUIET gstreamer-1.0 gstreamer-app-1.0)
    if(GSTREAMER_FOUND)
        set(HAVE_GSTREAMER ON)
    endif()
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== PRIVES build configuration ===")
message(STATUS "  OpenCV     : YES")
message(STATUS "  FFTW3      : ${HAVE_FFTW3}")
message(STATUS "  Iir1       : ${HAVE_IIR1}")
message(STATUS "  HailoRT    : ${HAVE_HAILORT}")
message(STATUS "  GStreamer   : ${HAVE_GSTREAMER}")
message(STATUS "")

# ---------------------------------------------------------------------------
# Unit tests (portable - always built)
# ---------------------------------------------------------------------------

# test_anonymizer - requires only OpenCV
add_executable(test_anonymizer
    tests/test_anonymizer.cpp
    anonymizer/Anonymizer.cpp
)
target_link_libraries(test_anonymizer PRIVATE ${OpenCV_LIBS} Catch2::Catch2WithMain)
catch_discover_tests(test_anonymizer)

# test_pos - requires only OpenCV
add_executable(test_pos
    tests/test_pos.cpp
    extraction/PosExtractor.cpp
)
target_link_libraries(test_pos PRIVATE ${OpenCV_LIBS} Catch2::Catch2WithMain)
catch_discover_tests(test_pos)

# test_detector - requires only OpenCV (uses DetectionUtils, no HailoRT)
add_executable(test_detector
    tests/test_detector.cpp
    detection/DetectionUtils.cpp
)
target_link_libraries(test_detector PRIVATE ${OpenCV_LIBS} Catch2::Catch2WithMain)
catch_discover_tests(test_detector)

# ---------------------------------------------------------------------------
# Unit tests (require additional libraries)
# ---------------------------------------------------------------------------

# test_estimation - requires FFTW3 + Iir1
if(HAVE_FFTW3 AND HAVE_IIR1)
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
        Catch2::Catch2WithMain
        pthread
    )
    catch_discover_tests(test_estimation)
else()
    message(STATUS "  [SKIP] test_estimation (missing FFTW3 or Iir1)")
endif()

# ---------------------------------------------------------------------------
# Demo executable (requires HailoRT + GStreamer - hardware only)
# ---------------------------------------------------------------------------
if(HAVE_HAILORT AND HAVE_GSTREAMER)
    add_executable(facedetection_demo
        demo/facedetection_demo.cpp
        detection/FaceDetectorSCRFD.cpp
        detection/DetectionUtils.cpp
    )
    target_include_directories(facedetection_demo PRIVATE
        ${GSTREAMER_INCLUDE_DIRS}
        ${HAILORT_INCLUDE_DIR}
    )
    target_link_libraries(facedetection_demo PRIVATE
        ${OpenCV_LIBS}
        ${GSTREAMER_LIBRARIES}
        ${HAILORT_LIB}
        pthread
    )
else()
    message(STATUS "  [SKIP] facedetection_demo (missing HailoRT or GStreamer)")
endif()

# ---------------------------------------------------------------------------
# PRIVES main pipeline (requires ALL hardware/library dependencies)
# ---------------------------------------------------------------------------
if(HAVE_HAILORT AND HAVE_GSTREAMER AND HAVE_FFTW3 AND HAVE_IIR1)
    add_executable(prives
        main.cpp
        detection/FaceDetectorSCRFD.cpp
        detection/DetectionUtils.cpp
        anonymizer/Anonymizer.cpp
        extraction/PosExtractor.cpp
        estimation/estimation.cpp
    )
    target_include_directories(prives PRIVATE
        ${GSTREAMER_INCLUDE_DIRS}
        ${HAILORT_INCLUDE_DIR}
        ${FFTW3_INCLUDE_DIRS}
        ${IIR1_INCLUDE_DIR}
    )
    target_link_libraries(prives PRIVATE
        ${OpenCV_LIBS}
        ${GSTREAMER_LIBRARIES}
        ${HAILORT_LIB}
        ${FFTW3_LIBRARIES}
        ${IIR1_LIB}
        pthread
    )
else()
    message(STATUS "  [SKIP] prives (missing HailoRT, GStreamer, FFTW3, or Iir1)")
endif()
