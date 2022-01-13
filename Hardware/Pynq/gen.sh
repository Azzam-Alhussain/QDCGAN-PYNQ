#!/bin/bash

# to be compiled on pynq
g++ -g -std=c++11 -pthread -O3 -fPIC -shared layer_loader.cpp -o layer_loader.so -lcma

