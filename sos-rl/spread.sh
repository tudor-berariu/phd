#!/bin/bash

rm -rf experiments
mkdir experiments

for ag in `seq 1 8`; do
    for s1 in A O; do
        for s2 in E NE; do
            DIR=$ag"_"$s1"_"$s2
            mkdir experiments/$DIR
            cp Scenarios.jl experiments/$DIR/
            cp ReinforcementLearning.jl experiments/$DIR/
            cp Miners.jl experiments/$DIR/
            cp run_ql.jl experiments/$DIR/
            cd experiments/$DIR
            sed -i 's/usr\/local\/bin\/julia/usr\/bin\/julia/' run_ql.jl
            # Correct number of agents
            sed -i "s/const AGENTS_NO        = 8;/const AGENTS_NO        = "$ag";/" Miners.jl
            # Correct strategies
            if [ "$s1" == "O" ]; then
                sed -i "s/const LEARNS_ONE = false;/const LEARNS_ONE = true;/" ReinforcementLearning.jl
            fi
            if [ "$s2" == "E" ]; then
                sed -i "s/const EXCHANGE = false;/const EXCHANGE = true;/" ReinforcementLearning.jl
            fi
            mkdir results
            cd ../..
        done
    done
done
