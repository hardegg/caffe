cmake_minimum_required(VERSION 2.8)

project( FaceDetection )

set(CMAKE_BUILD_TYPE Debug)
find_package( OpenCV REQUIRED )

add_executable( FaceDetection main.cpp )
target_link_libraries( FaceDetection ${OpenCV_LIBS} )
