#!/bin/bash

rm images/render*.png
cd build
make
./gpu/main_gpu < ../config.txt