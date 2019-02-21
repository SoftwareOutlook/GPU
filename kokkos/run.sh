#!/bin/bash

export OMP_PROC_BIND=spread
./test $1 $2
