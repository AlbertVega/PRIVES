cmake_minimum_required(VERSION 3.16)
project(PRIVES LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar OpenCV instalado en el sistema
find_package(OpenCV REQUIRED)

# Includes y dependencias
include_directories(${OpenCV_INCLUDE_DIRS})

# Definir el ejecutable de prueba con todas sus fuentes
add_executable(test_anonymizer 
    tests/test_anonymizer.cpp 
    anonymizer/Anonymizer.cpp
)

# Enlazar con OpenCV
target_link_libraries(test_anonymizer PRIVATE ${OpenCV_LIBS})