cmake_minimum_required(VERSION 2.8)

# MPI
find_package(MPI REQUIRED)
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

# Compiler settings
SET(CMAKE_C_FLAGS "-Wall -O3 -std=c99")

# Project settings
include_directories(
  include
)

# Labs
set(LABS 2d_integrate 3d_integrate sum matrix_multiply tridiagonal)

FOREACH(NAME ${LABS})
  set(SOURCES src/${NAME}.c)
  add_executable(${NAME} ${SOURCES})
ENDFOREACH(NAME)
