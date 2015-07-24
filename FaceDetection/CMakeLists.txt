cmake_minimum_required(VERSION 2.8)

project( FaceDetection )

find_package( OpenCV REQUIRED )

add_executable( FaceDetection main.cpp )
target_link_libraries( FaceDetection ${OpenCV_LIBS} )

project(GenInitNeg)

FIND_PACKAGE( Boost 1.58 COMPONENTS filesystem system thread program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
add_executable( GenInitNeg GenerateInitNeg.cpp utilities_common.cpp)
target_link_libraries( GenInitNeg ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
