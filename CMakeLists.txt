cmake_minimum_required(VERSION 3.1)

# Enable C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# allwpilib uses deprecated functions like sprintf and they compile it with werror
# mute deprecated functions warnings
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wno-deprecated-declarations")

# allwpilib options https://github.com/wpilibsuite/allwpilib/blob/v2023.4.2/README-CMAKE.md#build-options
set(WITH_JAVA OFF CACHE BOOL "Turn off Java" FORCE)
set(WITH_SHARED_LIBS OFF CACHE BOOL "Turn off shared libs" FORCE)
set(WITH_CSCORE OFF CACHE BOOL "Turn off CSCore" FORCE)
set(WITH_NTCORE ON CACHE BOOL "Turn on NTCore" FORCE)
set(WITH_WPIMATH OFF CACHE BOOL "Turn off wpimath" FORCE)
set(WITH_WPILIB OFF CACHE BOOL "Turn off wpilib" FORCE)
set(WITH_EXAMPLES OFF CACHE BOOL "Turn off examples" FORCE)
set(WITH_TESTS OFF CACHE BOOL "Turn off tests" FORCE)
set(WITH_GUI OFF CACHE BOOL "Turn off gui" FORCE)
set(WITH_SIMULATION_MODULES OFF CACHE BOOL "Turn off simulation modules" FORCE)

# Add depthai-core and allwpilib dependencies
# add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/depthai-core EXCLUDE_FROM_ALL)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/allwpilib EXCLUDE_FROM_ALL)

project(Detection)

# include(CTest)
# enable_testing()

find_package(OpenCV 4 REQUIRED)
find_package(Threads REQUIRED)

set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME     OFF)
find_package(Boost 1.74.0 REQUIRED)

set(WPILIB_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/allwpilib/ntcore/src/main/native/include)

# MainReactor
include_directories(./ ../MainReactor ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${WPILIB_INCLUDE_DIRS_DIRS})
add_library(shared_mat ../MainReactor/shared_mat/shared_mat.cpp)
add_library(camera ../MainReactor/camera/camera.cpp ../MainReactor/camera/video_capture.cpp)

add_executable(detection detection.cpp)
target_link_libraries(detection shared_mat camera ${Boost_LIBRARIES} ${OpenCV_LIBS} ntcore)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
