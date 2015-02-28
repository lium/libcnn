#cmake_minimum_required(VERSION 2.6)

#project(unit_test)

add_definitions(-std=c++0x -g -fopenmp -O3)
 
#Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Locate Eigen3
find_path(EIGEN_INCLUDE_DIR Eigen/Core
 /usr/local/include
)
include_directories(${EIGEN_INCLUDE_DIR})

# Locate OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#include_directories("../include")
include_directories(${PROJECT_SOURCE_DIR}/include)
set(SRC ../src/conv_layer.cpp)

# Link runTests with what we want to test and the GTest and pthread library
add_executable(runGoogleTests 
test_util_fun.cpp 
${SRC})

#add_executable(runTests main.cpp)
#target_link_libraries(unit_test1)

target_link_libraries(runGoogleTests ${GTEST_LIBRARIES} pthread)
