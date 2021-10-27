project(matrix CXX CUDA)
cmake_minimum_required(VERSION 3.16)

find_package(CUDA)
find_package(MPI)

add_executable(matrix main.cpp matrix.cu)

target_link_libraries(matrix ${MPI_CXX_LIBRARIES})
target_link_libraries(matrix ${CUDA_LIBRARIES})
include_directories(
	${PROJECT_SOURCE_DIR}
	${CUDA_INCLUDE_DIRS}
	${MPI_CXX_INCLUDE_PATH}
)