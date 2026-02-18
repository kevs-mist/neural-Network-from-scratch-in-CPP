cmake_minimum_required(VERSION 4.1)
project(NN)

set(CMAKE_CXX_STANDARD 20)

add_executable(NN main.cpp
        DataLoader.cpp)
