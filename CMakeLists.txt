cmake_minimum_required(VERSION 3.5)
project(Puffinn CXX)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Wno-noexcept-type -Wno-implicit-fallthrough -Wno-unused-function -march=native -O3 -g")

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories("include")

add_executable(GloveExample "examples/glove.cpp")
if (OpenMP_FOUND)
    target_link_libraries(GloveExample ${OpenMP_CXX_LIBRARIES})
endif()

include_directories("test/include")
add_executable(Test "test/main.cpp" "test/code.cpp")


find_package(HDF5 COMPONENTS CXX)
if (HDF5_FOUND)
    include(FetchContent)

    FetchContent_Declare(

        nanobench

        GIT_REPOSITORY https://github.com/martinus/nanobench.git

        GIT_TAG v4.1.0

        GIT_SHALLOW TRUE)


    FetchContent_MakeAvailable(nanobench)
    
    add_executable(Quick "experiments/main.cpp")
    target_include_directories(Quick PUBLIC "experiments/include")
    target_include_directories(Quick PUBLIC "${HDF5_CXX_INCLUDE_DIRS}")
    target_link_libraries(Quick ${HDF5_CXX_LIBRARIES})
    target_link_libraries(Quick nanobench)
endif()

if (OpenMP_FOUND)
    target_link_libraries(Test ${OpenMP_CXX_LIBRARIES})
endif()

add_subdirectory(docs)

