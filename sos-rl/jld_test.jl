using HDF5, JLD;

Qs = Array(Dict{Array{Uint8,1}, Dict{Uint8, Float64}}, 10);

JLD.save("qs.jld", "Qs", Qs);

Qs2 = JLD.load("qs.jld", "Qs");
