#!/usr/local/bin/julia

using Miners;
using ReinforcementLearning;

@time learn(Gatherers, 1000, 5000);
