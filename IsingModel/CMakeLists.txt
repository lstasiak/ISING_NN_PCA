cmake_minimum_required(VERSION 3.19)
project(Ising2021)

set(CMAKE_CXX_STANDARD 20)
add_executable(Ising2021 main.cpp Timer.h Utils.cpp Utils.h Models.cpp Models.h)

include_directories(includes/pcg_random_generator)