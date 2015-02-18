#!/usr/local/bin/julia

using Miners;
using ReinforcementLearning;

@time learn(Gatherers, 5000, 10000);
