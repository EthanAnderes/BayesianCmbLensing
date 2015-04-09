
# Introduction

This repository contains code for generating flat sky lensed CMB simulations and a Gibbs algorithm for generating posterior samples of the lensing potential and the noise-free lensed CMB conditional on the data. The algorithm is described in the paper "Bayesian estimates of CMB gravitational lensing" by Anderes, Wandelt and Guilhem. 

Requirements: 

 * Julia v0.3+ 
 * packages PyCall and PyPlot.
 * Python 2.7 with numpy, scipy and matplotlib

Note: to install the Julia packages execute the following commands at the Julia command line:

```julia
julia> Pkg.add("PyCall")
julia> Pkg.add("PyPlot")
```


# Generating a lensed CMB simulation and a Gibbs run

The script `scripts/scriptParallel.jl` generates a lensed CMB simulation and multiple Gibbs chains, in parallel, for sampling from the posterior on the lensing potential and the noise-free lensed CMB. Parameters of the run can be set within the script. To run this script launch Julia within the main directory:

```julia
$ julia
julia> addprocs(10)   # starts 10 parallel workers
julia> include("scripts/scriptParallel.jl")
```

The data will be saved to the directory `simulations/` with an serial number identifier that corresponds to the seed used to generate the simulated data. Currently the data is saved in csv files for cross-platform portability. However, this format is disk memory intensive so keep an eye on disk usage for long runs.


