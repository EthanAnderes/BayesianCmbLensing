"""
Generate figure 3 which shows embedding of the lensed grid from Section 6.
===================================================== """


"""
Load packages. Set prelim grids etc.
----------------------------------------------------- """
srcpath  = "/Users/ethananderes/Dropbox/BayesLense/src/"
savepath = "/Users/ethananderes/Google\ Drive/BayesLenseRev1/paper_rev2/"

include(srcpath*"Interp.jl")
include(srcpath*"cmb.jl")
include(srcpath*"fft.jl")

using PyPlot
using Interp

seed = 12
srand(seed)

par = setpar(0.35, 2^11, 0, 0, 3000, 1000, srcpath)
x,y = meshgrid(1:7,1:7)



"""
Generate figure3a.pdf
----------------------------------------------------- """
phik   = fft2(randn(size(par.grd.x))./par.grd.deltx, par) .* sqrt(par.cPP)
phidx1 = ifft2r(im .* par.grd.k1 .* phik, par)
phidx2 = ifft2r(im .* par.grd.k2 .* phik, par)    

grng = linspace(1, 100, size(x,1)) |> int
fac = 1.9e3
xnew = x + fac*phidx1[grng,grng] .- mean(fac*phidx1[grng,grng])
ynew = y + fac*phidx2[grng,grng] .- mean(fac*phidx2[grng,grng])
plt.scatter(xnew, ynew, s = 58, facecolors="k")
plt.annotate(L"$(x+\nabla\phi(x),\, data(x))$",
            xy=(xnew[6,6], ynew[6,6]), 
            xytext=(4.5, 7.2), 
            fontsize=18,
            arrowprops={:arrowstyle=>"->"}
)
axis("tight")
axis("off")
savefig(joinpath(savepath, "figure3a.pdf"), dpi=300, bbox_inches="tight", pad_inches=0) 



"""
Generate figure3b.pdf
----------------------------------------------------- """
a = linspace(2,8, 25) - ones(1,25) - .2
at = a' +.1
xnewround = zeros(size(xnew))
ynewround = zeros(size(ynew))
for k in 1:length(xnewround)
    indd = indmin( (xnew[k] .- a).^2 + (ynew[k] .- at).^2)
    xnewround[k] = a[indd]
    ynewround[k] = at[indd]
end
plt.scatter(xnewround, ynewround, s = 58, facecolors="k")
plt.scatter(a, at, facecolors="none", edgecolors="k", s = 58, lw = .95) #, alpha = .95)
plt.annotate(L"$(y,\, T(y) + \tilde n(y))$",
  xy=(xnewround[6,6], ynewround[6,6]), 
  xytext= (4.5, 7.2),  
  fontsize = 18,
  arrowprops={:arrowstyle => "->"}
)
plt.annotate(L"$(y,\, T(y) + \tilde n(y))$",
  xy=(a[22,23], at[22,24]), 
  xytext = (4.5, 7.2),  
  fontsize = 18,
  arrowprops={:arrowstyle => "->"}
)
axis("tight")
axis("off")
savefig(joinpath(savepath, "figure3b.pdf"), dpi=300, bbox_inches="tight") 












