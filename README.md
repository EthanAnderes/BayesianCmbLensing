To run a script in examples start julia and

```julia
julia> include("examples/script.jl")
```


* `run.jl` contains all the launch code for initializing and starting the chain to approximately sample from the posterior on the lensing potential. The second line of `run.jl` specifies the savepath: where the results should be saved to. I currently have this set outside of the working directory (so the data isn't loaded up to github).

* `clipboard.jl` is a file which simply contains some temporary code that was used while experimenting with the data. I don't necessary want to delete it since I sometimes find it useful to return back to the code for testing, etc.


