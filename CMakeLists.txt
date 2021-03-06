cmake_minimum_required(VERSION 2.8)
project(experiments)

set(VEXCL_ROOT $ENV{VEXCL_ROOT} CACHE STRING "VexCL root")
include_directories( ${VEXCL_ROOT} )

#----------------------------------------------------------------------------
# Show generated kernels
#----------------------------------------------------------------------------
option(VEXCL_SHOW_KERNELS "Show generated kernels")
if (VEXCL_SHOW_KERNELS)
    add_definitions(-DVEXCL_SHOW_KERNELS)
endif (VEXCL_SHOW_KERNELS)

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
find_package(Boost COMPONENTS system filesystem)

include_directories( ${Boost_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Find OpenCL
#----------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(OpenCL REQUIRED)

include_directories( ${OPENCL_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Enable C++11 support, set compilation flags
#----------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -Wclobbered -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -Wunused-parameter -Wunused-but-set-parameter")
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

    option(USE_LIBCPP "Use libc++ with Clang" OFF)
    if (USE_LIBCPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif (USE_LIBCPP)
endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

#----------------------------------------------------------------------------
function(add_experiment EXP)
    add_executable(${EXP} ${EXP}.cpp)
    target_link_libraries(${EXP} ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
endfunction()

add_experiment(static_vs_dynamic_reductor)
add_experiment(slice_reductor_perf)
add_experiment(sort-by-key-boost-compute)
add_experiment(sort-by-key-atomic)
add_experiment(matrix-assembly-cl)
add_experiment(matrix-assembly-vexcl)
add_experiment(vex_sfinae)
add_experiment(transpose)
add_experiment(dot_vs_reshape_n_reduce)

find_package(CUDA)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "-arch=sm_13")

cuda_add_executable(sort-by-key-thrust
    sort-by-key-thrust.cpp
    sort-by-key-thrust.cu
    )
target_link_libraries(sort-by-key-thrust ${Boost_LIBRARIES} ${CUDA_CUDA_LIBRARY})

cuda_add_executable(matrix-assembly-thrust
    matrix-assembly-thrust.cpp
    sort-by-key-thrust.cu
    )
target_link_libraries(matrix-assembly-thrust ${Boost_LIBRARIES} ${CUDA_CUDA_LIBRARY})
