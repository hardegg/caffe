cmake_minimum_required(VERSION 2.8)

project( test )
find_package( OpenCV REQUIRED )
add_executable( test main.cpp )
target_link_libraries( test ${OpenCV_LIBS} )

project(GenInitNeg)
find_package( Boost 1.58 COMPONENTS filesystem system thread program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
add_executable( GenInitNeg GenerateInitNeg.cpp utilities_common.cpp)
target_link_libraries( GenInitNeg ${OpenCV_LIBS} ${Boost_LIBRARIES}) 

project(FaceDetection)
find_package( OpenCV REQUIRED )
find_package( Boost 1.58 COMPONENTS filesystem system thread program_options REQUIRED )
add_executable(FaceDetection FaceDetection.cpp FaceClassifier.cpp)
target_link_libraries(FaceDetection ${Caffe_LINK} ${OpenCV_LIBS} ${Boost_LIBRARIES})
