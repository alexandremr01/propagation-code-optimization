#!/bin/bash

# Usage: ./all_threads.sh

export KMP_AFFINITY=balanced
export CONFIG_EXE_NAME=all_nthreads_test.exe

make -C iso3dfd-st7/ Olevel=-O2 simd=avx last

touch all_threads.txt
for nthreads in 25 26 27 29 30 31
do
    echo "nthreads=$nthreads" | tee -a all_threads.txt
    for i in {1..5}
    do
        /usr/bin/mpirun -np 1 -map-by ppr:1:node:PE=16 iso3dfd-st7/bin/$CONFIG_EXE_NAME 512 512 512 $nthreads 100 512 4 10 | egrep 'throughput:' >> all_threads.txt
    done
done