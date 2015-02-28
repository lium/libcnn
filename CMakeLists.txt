cmake_minimum_required(VERSION 2.6)

project(libcnn++)

add_definitions(-std=c++0x -g -fopenmp -O3)
 
# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Locate Eigen3
find_path(EIGEN_INCLUDE_DIR Eigen/Core
 /usr/local/include
 /usr/include
)
include_directories(${EIGEN_INCLUDE_DIR})

find_package(OpenCV REQUIRED)

add_subdirectory(test)

# Locate OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories("${PROJECT_SOURCE_DIR}/include")
set(SRC
${CMAKE_SOURCE_DIR}/src/conv_layer.cpp
${CMAKE_SOURCE_DIR}/src/load_data.cpp
${CMAKE_SOURCE_DIR}/src/softmax_class.cpp
${CMAKE_SOURCE_DIR}/src/pool.cpp
${CMAKE_SOURCE_DIR}/src/init_params.cpp
${CMAKE_SOURCE_DIR}/src/preprocess.cpp
${CMAKE_SOURCE_DIR}/src/rand_initialize.cpp
${CMAKE_SOURCE_DIR}/src/save_params.cpp
${CMAKE_SOURCE_DIR}/src/online_test.cpp
${CMAKE_SOURCE_DIR}/src/save_load.cpp
#${CMAKE_SOURCE_DIR}/src/train.cpp)
${CMAKE_SOURCE_DIR}/src/validate.cpp)


#set(TRAIN ${CMAKE_SOURCE_DIR}/src/train.cpp)
#set(EVAL  ${CMAKE_SOURCE_DIR}/src/evaluate.cpp)

include_directories(/usr/local/include/yaml-cpp)
link_directories(/usr/local/lib/)

# Link runTests with what we want to test and the GTest and pthread library
add_executable(train ${SRC})
#add_executable(evalTests ${SRC} ${EVAL})
#target_link_libraries(runTests yaml-cpp)

target_link_libraries(train ${GTEST_LIBRARIES} ${OpenCV_LIBS} pthread)
