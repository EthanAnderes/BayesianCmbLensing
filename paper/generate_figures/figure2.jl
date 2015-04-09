"""
Generate figure 2 which the difference between anti-lensing and inverse lensing
===================================================== """


"""
Load packages. Set grid etc.
----------------------------------------------------- """
srcpath  = "/Users/ethananderes/Dropbox/BayesLense/src/"
savepath = "/Users/ethananderes/Dropbox/BayesLense/paper/"

include(srcpath*"Interp.jl")
include(srcpath*"cmb.jl")
include(srcpath*"fft.jl")

using PyPlot, PyCall, Interp
@pyimport scipy.interpolate as scii

seed = 1
srand(seed)

parlr = setpar(2.0, 2^9,  0, 0, 3000, 600, srcpath);
parhr = setpar(0.5, 2^11, 0, 0, 3000, 600, srcpath);



""" Simulate the lensing potential, define functions
-----------------------------------------------"""
phik   = fft2(randn(size(parhr.grd.x))./parhr.grd.deltx, parhr) .* sqrt(parhr.cPP)
phix   = ifft2r(phik, parhr.grd.deltk)
phidx1 = ifft2r(im .* parhr.grd.k1 .* phik, parhr)
phidx2 = ifft2r(im .* parhr.grd.k2 .* phik, parhr)    

function griddata(x::Matrix, y::Matrix, z::Matrix, xi::Matrix,yi::Matrix)
	xpd, ypd = Interp.perodic_padd_xy(x, y, 0.1)
	zpd = Interp.perodic_padd_z(z, 0.1)
	points = [xpd[:] ypd[:]]
	# grid = (vec(xi[1,:]), vec(yi[:,1]))
	grid = (xi, yi)
	zi = scii.griddata(points, zpd[:], grid, method = "cubic")
end

lensex = parhr.grd.x + phidx1
lensey = parhr.grd.y + phidx2
antix_dsp = griddata(
	lensex, 
	lensey,
	-phidx1, 
	parlr.grd.x, 
	parlr.grd.y
)
antiy_dsp = griddata(
	lensex, 
	lensey,
	-phidx2, 
	parlr.grd.x, 
	parlr.grd.y
)

function helmholtz(ax::Matrix, bx::Matrix, par::SpectrumGrids)
	# (ax,bx) is the vector field defined in pixel space
	ak, bk = fft2(ax, par), fft2(bx, par)
	k₁, k₂ = par.grd.k1, par.grd.k2
	adk = (k₁./k₂) .* ((k₁./k₂) .* ak + bk) ./ ((k₁./k₂).^2 .+ 1) # curl free pat
	ack = ak - adk 
	divk = adk ./ ( im * k₁)
	crlk = ack ./ (-im * k₂) 
	divk[par.grd.k1 .== 0.0] = bk[par.grd.k1 .== 0.0] ./ ( im * k₂[par.grd.k1 .== 0.0])
	divk[par.grd.k2 .== 0.0] = ak[par.grd.k2 .== 0.0] ./ ( im * k₁[par.grd.k2 .== 0.0])
	crlk[par.grd.k1 .== 0.0] = ak[par.grd.k1 .== 0.0] ./ (-im * k₂[par.grd.k1 .== 0.0])
	crlk[par.grd.k2 .== 0.0] = bk[par.grd.k2 .== 0.0] ./ ( im * k₁[par.grd.k2 .== 0.0])
	divk[par.grd.r .<= 0.0] = 0.0
	crlk[par.grd.r .<= 0.0] = 0.0
	divx, crlx = ifft2r(divk, par), ifft2r(crlk, par)
	divx, divk, crlx, crlk
end
divx, divk, crlx, crlk = helmholtz(antix_dsp, antiy_dsp, parlr)




"""
figure2a.pdf
-----------------------------------"""
imshow(
  -phix[1:4:end, 1:4:end],
  interpolation = "nearest",
  origin="lower",
  extent=(180/pi)*[minimum(parlr.grd.x), maximum(parlr.grd.x),minimum(parlr.grd.x), maximum(parlr.grd.x)]
)
title(L"$-\phi$")
xlabel("degrees")
ylabel("degrees")
colorbar(format="%.e")
savefig(joinpath(savepath, "figure2a.pdf"), dpi=300, bbox_inches="tight", pad_inches=0) 



"""
figure2b.pdf
-----------------------------------"""
imshow(
  divx,
  interpolation = "nearest",
  origin="lower",
  extent=(180/pi)*[minimum(parlr.grd.x), maximum(parlr.grd.x),minimum(parlr.grd.x), maximum(parlr.grd.x)]
)
title(L"$-\phi^{inv}$")
xlabel("degrees")
ylabel("degrees")
colorbar(format="%.e")
savefig(joinpath(savepath, "figure2b.pdf"), dpi=300, bbox_inches="tight", pad_inches=0) 




"""
figure2c.pdf
-----------------------------------"""
imshow(
  divx + phix[1:4:end, 1:4:end],
  interpolation = "nearest",
  origin="lower",
  extent=(180/pi)*[minimum(parlr.grd.x), maximum(parlr.grd.x),minimum(parlr.grd.x), maximum(parlr.grd.x)]
)
title(L"$\phi-\phi^{inv}$")
xlabel("degrees")
ylabel("degrees")
colorbar(format="%.e")
savefig(joinpath(savepath, "figure2c.pdf"), dpi=300, bbox_inches="tight", pad_inches=0) 




"""
figure2d.pdf
-----------------------------------"""
imshow(
  crlx,
  vmin = -1.2e-8, vmax = 1.2e-8,
  interpolation = "nearest",
  origin="lower",
  extent=(180/pi)*[minimum(parlr.grd.x), maximum(parlr.grd.x),minimum(parlr.grd.x), maximum(parlr.grd.x)]
)
title(L"$-\psi^{inv}$")
xlabel("degrees")
ylabel("degrees")
colorbar(format="%.e")
savefig(joinpath(savepath, "figure2d.pdf"), dpi=300, bbox_inches="tight", pad_inches=0) 



