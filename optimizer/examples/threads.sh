#!/bin/bash

# Usage: ./threads.sh --n 16

export KMP_AFFINITY=balanced
export CONFIG_EXE_NAME=nthreads_test.exe

nthreads=$2

make -C iso3dfd-st7/ Olevel=-O2 simd=avx last

for i in {1..5}
do
    /usr/bin/mpirun -np 1 -map-by ppr:1:node:PE=16 iso3dfd-st7/bin/$CONFIG_EXE_NAME 512 512 512 $nthreads 100 512 4 10
done