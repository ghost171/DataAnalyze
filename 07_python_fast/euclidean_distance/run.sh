#!/usr/bin/env bash

set -e

gcc -O3 -std=gnu17 -c -o euclidean_distance.o euclidean_distance.c
gcc -shared -o euclidean_distance.so euclidean_distance.o
