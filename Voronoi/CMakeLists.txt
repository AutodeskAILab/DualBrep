cmake_minimum_required(VERSION 3.10)
project(Make-A-BRep)

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/external/GTE/GTE")
add_subdirectory(calculate_voronoi)