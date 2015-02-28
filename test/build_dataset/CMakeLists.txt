cmake_minimum_required(VERSION 2.8)
project( build_dataset )
find_package( OpenCV REQUIRED )
add_executable( build_dataset build_dataset.cpp )
target_link_libraries( build_dataset ${OpenCV_LIBS} )

