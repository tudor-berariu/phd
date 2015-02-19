#!/bin/bash

for ag in `seq 1 8`; do
    for s1 in A O; do
        for s2 in E NE; do
            DIR=$ag"_"$s1"_"$s2
            touch experiments/$DIR/do_save
        done
    done
done

