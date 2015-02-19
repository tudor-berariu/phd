#!/bin/bash

for ag in `seq 1 8`; do
    for s1 in A O; do
        for s2 in E NE; do
            DIR=$ag"_"$s1"_"$s2
            cd experiments/$DIR
            if [ -f output.log ]; then
                echo -n $ag" : "$s1" : "$s2" : "
                tail -n 1 output.log
            fi
            cd ../..
        done
    done
done
