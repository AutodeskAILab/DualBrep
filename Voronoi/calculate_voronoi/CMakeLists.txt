cmake_minimum_required (VERSION 3.10)
PROJECT(calculate_voronoi)

set(CMAKE_CXX_STANDARD 11)

FILE(GLOB SOURCE_FILE *.cpp *.h)

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(glog CONFIG REQUIRED)
find_package(OpenCASCADE CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
#set(GEOGRAM_INSTALL_PREFIX ${ROOT_PATH}/external/geogram/build/install/)
#include(${ROOT_PATH}/external/geogram/build/install/lib/cmake/modules/FindGeogram.cmake)
find_package(Geogram CONFIG REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILE})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCASCADE_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ../external/GTE/GTE)
target_include_directories(${PROJECT_NAME} PRIVATE ../external/vcglib)

target_link_libraries(${PROJECT_NAME} PRIVATE Geogram::geogram)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCASCADE_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE glog::glog)
target_link_libraries(${PROJECT_NAME} PRIVATE CGAL::CGAL)
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::boost Boost::filesystem)