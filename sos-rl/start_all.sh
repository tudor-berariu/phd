#!/bin/bash

ag=$1
for s1 in A O; do
    for s2 in E NE; do
        DIR=$ag"_"$s1"_"$s2
        cd experiments/$DIR
        nohup ./run_ql.jl >output.log 2>&1 &
        cd ../..
    done
done
