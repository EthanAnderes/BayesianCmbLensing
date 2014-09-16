
# Introduction




Requirements: 

 * Julia v0.3+ 
 * packages PyCall and PyPlot.
 * Python 2.7 with numpy, scipy and matplotlib

Note: to install the Julia packages execute the following commands at the Julia command line:

```julia
julia> Pkg.add("PyCall")
julia> Pkg.add("PyPlot")
```



# Generating a lensed CMB simulation and run gibbs chain

This script generates multiple simulated gibbs runs in parallel, all on the same simulated data.
There are two ways to run this file, both which requires launching Julia in the top diretory. 
The first way is to run the following command from the terminal

```
$ julia -p 10 scripts/scriptParallel.jl 
```

In the above command Julia is started with 10 independent workers and simulations are made in parallel.
You can change this to the number of cores/workers you want.
The second way is the launch the script from withing Julia:

```julia
$ julia
julia> addprocs(10)   # starts 10 parallel workers
julia> include("scripts/scriptParallel.jl")
```


# Generating Figures

To make figures you will need the PyPlot package installed.

This file generates figures based on the run from `scriptParallel`. 
Currently you need to edit the top of this script to set the directly to load data from and to set the 
directory to save the images to.

There are two ways to run this file, both which requires launching Julia in the top directory. 
The first way is to run the following command from the terminal

```
$ julia scripts/makeFigsParallel.jl 
```

The second way is the launch the script from withing Julia. When launching form Julia you can 
disable saving the figures to file and simply view them dynamically. 

```julia
$ julia
julia> include("scripts/makeFigsParallel.jl")
```