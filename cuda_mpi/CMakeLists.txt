project(cuda-mpi C CXX)
cmake_minimum_required(VERSION 3.16)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(MPI)

# mpi-hello
add_executable(mpi-hello mpiHelloWorld.cpp)
add_executable(send-recv sendRecv.cpp)
add_executable(random-walk randomWalk.cpp)
add_executable(bcast bcast.cpp)
if(MPI_FOUND)
	include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
	target_link_libraries(mpi-hello ${MPI_CXX_LIBRARIES})
	target_link_libraries(send-recv ${MPI_CXX_LIBRARIES})
	target_link_libraries(random-walk ${MPI_CXX_LIBRARIES})
	target_link_libraries(bcast ${MPI_CXX_LIBRARIES})
else(MPI_FOUND)
	message(FATAL_ERROR "MPI library not found")
endif(MPI_FOUND)
