"""
Generate figures 4, 5, .... which summarize a test run of the Gibbs.
===================================================== 
This file generates figures based on the run from `scriptParallel`. 
"""


"""
Load modules and set load and save links
------------------------------------------
"""
simdir   =  "/Users/ethananderes/Dropbox/BayesLense/simulations/scriptParallel_2657077506/" # contains the simulation
srcpath  =  "/Users/ethananderes/Dropbox/BayesLense/src/"
savepath = "/Users/ethananderes/Dropbox/BayesLense/paper/"

include(srcpath*"Interp.jl")
include(srcpath*"cmb.jl")
include(srcpath*"fft.jl")

using PyPlot, Interp



"""
Set the parameters  of the simulation
-----------------------------------------
"""
const percentNyqForC = 0.5 # used for T l_max
const numofparsForP  = 1500  # used for P l_max
const hrfactor = 2.0
const pixel_size_arcmin = 2.0
const n = 2.0^9
const beamFWHM = 0.0
const nugget_at_each_pixel = (4.0)^2
begin  #< ---- dependent run parameters
	local deltx =  pixel_size_arcmin * π / (180 * 60) #rads
	local period = deltx * n # side length in rads
	local deltk =  2 * π / period
	local nyq = (2 * π) / (2 * deltx)
	const maskupP  = √(deltk^2 * numofparsForP / π)  #l_max for for phi
	const maskupC  = min(9000.0, percentNyqForC * (2 * π) / (2 * pixel_size_arcmin * π / (180*60))) #l_max for for phi
end
const scale_grad =  2.0e-3
const scale_hmc  =  2.0e-3

d = 2
parlr = setpar(
	pixel_size_arcmin, 
	n,      
	beamFWHM, 
	nugget_at_each_pixel, 
	maskupC, 
	maskupP,
	srcpath
	);
dirac_0 = 1 / parlr.grd.deltk^d 
elp = parlr.ell[10:int(maskupP)] 
elt = parlr.ell[10:1000] 
cpl = parlr.CPell2d[10:int(maskupP)]
ctl = parlr.CTell2dLen[10:1000]
ct  = parlr.CTell2d[10:1000]
r2  = parlr.grd.r.^2
bin_mids_P = (parlr.grd.deltk*1.5):(parlr.grd.deltk):maskupP
bin_mids_T = (parlr.grd.deltk*1.5):(parlr.grd.deltk):1000




"""
Get Set the burning/thinning and detect the number of parallel runs
------------------------------------------
"""
krang = 551:100:2501  #  burning/thinning

jobs = Int[] # detect the number of parallel runs
if isdir("$simdir/job1")
	push!(jobs, 1)
else
	push!(jobs, 2)
	while isdir("$simdir/job$(jobs[end]+1)")
		push!(jobs,jobs[end]+1 )
	end
end
jobs





"""
Define functions which compute power and average over l - bins
------------------------------------------
"""
function binave(fk::Matrix, kmag::Matrix, bin_mids::Range)
fpwr = Array(Complex{Float64}, length(bin_mids))
fill!(fpwr, -1.0)
rtcuts  = collect(bin_mids +  step(bin_mids) / 2)  
lftcuts = collect(bin_mids -  step(bin_mids) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	for i in 1:length(bin_mids)
		ibin = lftcuts[i] .<= kmag .< rtcuts[i]
		fpwr[i] = fk[ibin] |> mean   
	end
	fpwr
end
function binpower(fk::Matrix, kmag::Matrix, bin_mids::Range)
	fpwr = Array(Float64, length(bin_mids))
	fill!(fpwr, -1.0)
	rtcuts  = collect(bin_mids +  step(bin_mids) / 2)  
	lftcuts = collect(bin_mids -  step(bin_mids) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	for i in 1:length(bin_mids)
		ibin = lftcuts[i] .<= kmag .< rtcuts[i]
		fpwr[i] = fk[ibin] |> abs2 |> mean   
	end
	fpwr
end




"""
Load the simulation runs
------------------------------------------"""
#-- simulation truths
phix   = readcsv("$simdir/phix.csv")
tildex = readcsv("$simdir/tildetx_lr.csv")
tx_lr  = readcsv("$simdir/tx_lr.csv")
phik   = fft2(phix, parlr)
tildek = fft2(tildex, parlr)
tk_lr  = fft2(tx_lr, parlr)
ytx    = readcsv("$simdir/ytx.csv")
qex    = readcsv("$simdir/qex_lr.csv")
x      = readcsv("$simdir/x.csv")
maskx    = readcsv("$simdir/maskvarx.csv")
maskbool = maskx .== Inf 

# initialize  cross correlation
phik_cov_smpl = Array{Float64,1}[]
tildek_cov_smpl = Array{Float64,1}[]
tk_cov_smpl = Array{Float64,1}[]

# initialize  spectral coverage
phik_pwr_smpl = Array{Float64,1}[]
tildek_pwr_smpl = Array{Float64,1}[]
tk_pwr_smpl = Array{Float64,1}[]

# initialize  1-d slices
propslice = 0.75 # btwn 0 and 1
phix_slice_samples = Array{Float64,1}[]
tildex_slice_samples = Array{Float64,1}[]
tx_slice_samples = Array{Float64,1}[]

# initialize  average 
phix_sum = zero(phix)
tildetx_lr_sum = zero(phix)
tx_lr_sum = zero(phix)

runs = 0
for r in jobs, k in krang
	if isfile("$simdir/job$r/phix_curr_$k.csv") 
	
		phix_curr = readcsv("$simdir/job$r/phix_curr_$k.csv")
		phik_curr = fft2(phix_curr, parlr)
		tildetx_curr = readcsv("$simdir/job$r/tildetx_lr_curr_$k.csv")
		tildetk_curr = fft2(tildetx_curr, parlr)
		tx_curr = readcsv("$simdir/job$r/tx_lr_curr_$k.csv")
		tk_curr = fft2(tx_curr, parlr)
	
		# spectral coverage
		push!(phik_pwr_smpl,   binpower(r2    .* phik_curr, parlr.grd.r, bin_mids_P))
		push!(tildek_pwr_smpl, binpower(√(r2) .* tildetk_curr, parlr.grd.r, bin_mids_T))
		push!(tk_pwr_smpl,     binpower(√(r2) .* tk_curr, parlr.grd.r, bin_mids_T))
	
		# cross correlation
		pnxt   = binave(  r2 .* phik_curr .* conj(phik), parlr.grd.r, bin_mids_P)
		pnxt ./= binpower(sqrt(r2) .* phik_curr, parlr.grd.r, bin_mids_P) |> sqrt
		pnxt ./= binpower(sqrt(r2) .* phik, parlr.grd.r, bin_mids_P) |> sqrt
		push!(phik_cov_smpl, real(pnxt))

		tnxt   = binave(  r2 .* tildetk_curr .* conj(tildek), parlr.grd.r, bin_mids_T)
		tnxt ./= binpower(sqrt(r2) .* tildetk_curr, parlr.grd.r, bin_mids_T) |> sqrt
		tnxt ./= binpower(sqrt(r2) .* tildek, parlr.grd.r, bin_mids_T) |> sqrt
		push!(tildek_cov_smpl, real(tnxt))

		tnxt   = binave(  r2 .* tk_curr .* conj(tk_lr), parlr.grd.r, bin_mids_T)
		tnxt ./= binpower(sqrt(r2) .* tk_curr, parlr.grd.r, bin_mids_T) |> sqrt
		tnxt ./= binpower(sqrt(r2) .* tk_lr, parlr.grd.r, bin_mids_T) |> sqrt
		push!(tk_cov_smpl, real(tnxt))
	
		# 1-d slices
		push!(phix_slice_samples, phix_curr[int(end*propslice),:][:])
		push!(tildex_slice_samples, tildetx_curr[int(end*propslice),:][:])
		push!(tx_slice_samples, tx_curr[int(end*propslice),:][:])
	
		# averages
		phix_sum += phix_curr
		tildetx_lr_sum += tildetx_curr
		tx_lr_sum += tx_curr

	runs += 1
	end
end

# 1-d slices
phix_slice_samples   = hcat(phix_slice_samples...)
tildex_slice_samples = hcat(tildex_slice_samples...)
tx_slice_samples     = hcat(tx_slice_samples...)
phix_true_slice      = phix[int(end*propslice),:][:]
tildex_true_slice    = tildex[int(end*propslice),:][:]
tx_true_slice        = tx_lr[int(end*propslice),:][:]
phix_sum_slice       = phix_sum[int(end*propslice),:][:]
tildetx_lr_sum_slice = tildetx_lr_sum[int(end*propslice),:][:]
tx_lr_sum_slice      = tx_lr_sum[int(end*propslice),:][:]
qex_slice  = qex[int(end*propslice),:][:]
x_slice    = x[int(end*propslice),:][:]
varx_slice = maskx[int(end*propslice),:][:]
isempty(x_slice[varx_slice.==Inf]) || (maskmin=minimum(x_slice[varx_slice.==Inf]); maskmax = maximum(x_slice[varx_slice.==Inf]))

# cross correlation
# squish to matrix
phik_cov_smpl   = hcat(phik_cov_smpl...)
tildek_cov_smpl = hcat(tildek_cov_smpl...)
tk_cov_smpl     = hcat(tk_cov_smpl...)
# then split by the rows
phb_cov          = Array{Float64,1}[vec(phik_cov_smpl[k,:])   for k=1:size(phik_cov_smpl,   1)]
tib_cov          = Array{Float64,1}[vec(tildek_cov_smpl[k,:]) for k=1:size(tildek_cov_smpl, 1)]
tib_unlensed_cov = Array{Float64,1}[vec(tk_cov_smpl[k,:]) for k=1:size(tk_cov_smpl, 1)]

# spectral coverage
# squish to matrix
phik_pwr_smpl   = hcat(phik_pwr_smpl...)
tildek_pwr_smpl = hcat(tildek_pwr_smpl...)
tk_pwr_smpl     = hcat(tk_pwr_smpl...)
# then split by the rows
phb_pwr          = Array{Float64,1}[phik_pwr_smpl[k,:][:]   for k=1:size(phik_pwr_smpl,   1)]
tib_pwr          = Array{Float64,1}[tildek_pwr_smpl[k,:][:] for k=1:size(tildek_pwr_smpl, 1)]
tib_unlensed_pwr = Array{Float64,1}[tk_pwr_smpl[k,:][:] for k=1:size(tk_pwr_smpl, 1)]

# here is the truth
phb_truth          = binpower(r2 .* phik, parlr.grd.r, bin_mids_P)
tib_truth          = binpower(sqrt(r2) .* tildek, parlr.grd.r, bin_mids_T)
tib_unlensed_truth = binpower(sqrt(r2) .* tk_lr, parlr.grd.r, bin_mids_T)




"""
Cooling schedule version 2
------------------------------------------
"""
function repstretch(start, stop, rep::Int, total::Int) 
    tdr = int(total / rep)
    return vec(linspace(start, stop, tdr)' .* ones(rep, tdr))
end
lcool = repstretch(2.0, 1.2maskupC, 75, 1000) |> int
λcool = Float64[]
d2x   = parlr.grd.deltx * parlr.grd.deltx
d2k   = parlr.grd.deltk * parlr.grd.deltk
delt0 = 1 / d2k
barNx = 0.99 * parlr.nugget_at_each_pixel
barNk = barNx * d2x # spectral density in fourier

ls = 1:4000
semilogy(ls, parlr.CTell2d[ls], color = "k",  linewidth=1.5, linestyle="-", label=L"$C_l^{TT}$")
annotate(L"$C_l^{TT}$",
        xy=(150, parlr.CTell2d[150]), xycoords="data",
        xytext=(35, 35), textcoords="offset points", fontsize=16,
        arrowprops={:arrowstyle=>"->"})
for cntr in [1, 100, 300, 500, 700, 900]
	λσ2dy = (cntr > 8000.0) ? barNk : max(barNk, parlr.CTell2d[lcool[cntr]]) 
	semilogy(ls, zeros(ls) + λσ2dy, color = "k",  linewidth=1.5, linestyle=":")
	annotate(latexstring("\$\\lambda_{$cntr}\\bar{\\sigma}^2dy\$"),
        xy=(lcool[cntr], λσ2dy), xycoords="data",
        xytext=(20, 7), textcoords="offset points", fontsize=16)
end
semilogy(ls, zeros(ls) + barNk, color = "k",  linewidth=1.5, linestyle="--", label="noise level")
annotate(L"homogeneous noise spectral density = $\bar{\sigma}^2dy$",
        xy=(1540, barNk), xycoords="data",
        xytext=(-100, -40), textcoords="offset points", fontsize=14,
        arrowprops={:arrowstyle=>"->"})
xlabel("wavenumber", fontsize=14)
ylabel("spectral density", fontsize=14)
xlim(0, 3700)
ylim(1e-7, 1e+4)
savefig(joinpath(savepath, "figure4.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.1)



"""
True phix
------------------------------------------
"""
imshow(
  phix, 
  interpolation = "nearest", 
  vmin=minimum(phix),
  vmax=maximum(phix), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
plt.xlabel("degrees")
plt.ylabel("degrees")
savefig(joinpath(savepath, "figure5a.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)




"""
Posterior mean for phix
------------------------------------------
"""
imshow(
  phix_sum/runs, 
  interpolation = "nearest", 
  vmin=minimum(phix),
  vmax=maximum(phix), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
plt.xlabel("degrees")
plt.ylabel("degrees")
savefig(joinpath(savepath, "figure5b.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)




"""
Quadratic estimate
------------------------------------------
"""
imshow(
	qex, 
  interpolation = "nearest", 
  vmin=minimum(phix),
  vmax=maximum(phix), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
plt.xlabel("degrees")
plt.ylabel("degrees")
savefig(joinpath(savepath, "figure5c.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)




"""
Data
------------------------------------------
"""
imshow(
  ytx, 
  interpolation = "nearest", 
  vmin=minimum(tildex),
  vmax=maximum(tildex), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
xlabel("degrees")
ylabel("degrees")
title("Data")
savefig(joinpath(savepath, "figure6a.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)





"""
Poster mean tx
------------------------------------------
"""
imshow(
  tx_lr_sum/runs, 
  interpolation = "nearest", 
  vmin=minimum(tildex),
  vmax=maximum(tildex), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
plt.xlabel("degrees")
plt.ylabel("degrees")
colorbar()
title(L"$E(T(x)|data)$")
savefig(joinpath(savepath, "figure6b.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)





"""
tx - tildex
------------------------------------------
"""
imshow(
  (tx_lr - tildex) .* (1- maskbool)  , 
  interpolation = "nearest", 
  vmin=0.8*minimum( (tx_lr - tildex) .* (1- maskbool) ),
  vmax=0.8*maximum( (tx_lr - tildex) .* (1- maskbool) ), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
plt.xlabel("degrees")
plt.ylabel("degrees")
colorbar(format="%.e")
title(L"$T(x) - \widetilde T(x)$")
savefig(joinpath(savepath, "figure6d.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)




"""
tx - E(tx|data)
------------------------------------------
"""
imshow(
   (tx_lr - tx_lr_sum/runs) .* (1- maskbool) , 
  interpolation = "nearest", 
  vmin=0.8*minimum( (tx_lr - tildex) .* (1- maskbool) ),
  vmax=0.8*maximum( (tx_lr - tildex) .* (1- maskbool) ), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
#title(latexstring("\$abs(T(x) - E(T(x)|data))\$"))
plt.xlabel("degrees")
plt.ylabel("degrees")
title(L"$T(x) - E(T(x)|data)$")
savefig(joinpath(savepath, "figure6c.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)



"""
Slices of phix
------------------------------------------
"""
#=
The slices are done at  degree mark 12.7
(180/pi) * parlr.grd.y[int(end*propslice),:]
=#
plot((180/pi)*x_slice, phix_slice_samples[:,1], color="blue", alpha=0.15, label=L"posterior samples of $\phi(x)$")
plot((180/pi)*x_slice, phix_slice_samples[:,2:end], color="blue", alpha=0.15)
plot((180/pi)*x_slice, phix_true_slice, color = "red",  linewidth=1.5, linestyle="-", label=L"simulation truth $\phi(x)$")
annotate("mask",
         xy=((180/pi)*maskmin, 0), xycoords="data",
         xytext=(-50, 50), textcoords="offset points", fontsize=16,
         arrowprops={:arrowstyle=>"->"})
xlabel("degrees")
axis("tight")
axvspan((180/pi)*maskmin, (180/pi)*maskmax,  facecolor="0.5", alpha=0.3, label="mask region")
legend(loc="upper left", fontsize = 12)
ticklabel_format(style="sci", axis="y", scilimits=(0,0))
savefig(joinpath(savepath, "figure7a.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)





"""
Slice of tx
------------------------------------------
"""
lper = 0.2
rper = 0.5
zoom = round(lper*length(x_slice)):round(rper*length(x_slice))
plot((180/pi)*x_slice[zoom], tx_slice_samples[zoom,1], color="blue", alpha=0.15, label=L"posterior samples $T(x)$")
plot((180/pi)*x_slice[zoom], tx_slice_samples[zoom,2:end], color="blue", alpha=0.15)
plot((180/pi)*x_slice[zoom], tx_true_slice[zoom], color = "r",  linewidth=1.4, linestyle="-", label=L"simulation truth $T(x)$")
axis("tight")
axvspan((180/pi)*maskmin, (180/pi)*maskmax,  facecolor="0.5", alpha=0.3, label="mask region")
annotate("mask", xy=((180/pi)*maskmin, 0), xycoords="data", xytext=(-50, -50), textcoords="offset points", fontsize=16, arrowprops={:arrowstyle=>"->"})
xlabel("degrees")
axis("tight")
legend(loc="upper left", fontsize = 12)
ticklabel_format(style="sci", axis="y", scilimits=(0,0))
savefig(joinpath(savepath, "figure7b.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)





"""
Spectral coverage for phi
------------------------------------------
"""
plot(elp, elp.^4 .* cpl / 4, "-k", label = L"$l^4C_l^{\phi\phi}/4$")
plot(bin_mids_P, phb_truth / dirac_0 / 4, "or", label = L"$l^4 |\phi_l|^2/(4\delta_0)$")
rtcuts  = collect(bin_mids_P +  step(bin_mids_P) / 2)  
lftcuts = collect(bin_mids_P -  step(bin_mids_P) / 2)  
lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
errorbar(
	bin_mids_P,
	map(median,  phb_pwr / dirac_0 / 4),
	xerr = Array{Float64,1}[bin_mids_P-lftcuts, rtcuts-bin_mids_P],  
	yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), phb_pwr / dirac_0 / 4), map(x-> quantile(x,0.975)-median(x), phb_pwr / dirac_0 / 4)],
	fmt="*b",
	label = L"95% posterior for $l^4 |\phi_l|^2/ (4 \delta_0)$"
)
plot(collect(bin_mids_P), zero(bin_mids_P), ":k")
xlabel("wavenumber")
legend()
ticklabel_format(style="sci", axis="y", scilimits=(0,0))
axis("tight")
savefig(joinpath(savepath, "figure8a.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)




"""
Spectral coverage for T
------------------------------------------
"""
plot(elt,  elt.^2 .* ct, "-k", label = L"$l^2C_l^{ TT}$")
plot(bin_mids_T, tib_unlensed_truth / dirac_0 , "or", label =  L"$l^2 |T_l|^2/ \delta_0$")
rtcuts  = collect(bin_mids_T +  step(bin_mids_T) / 2)  
lftcuts = collect(bin_mids_T -  step(bin_mids_T) / 2)  
lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
errorbar(
	bin_mids_T,
	map(median,  tib_unlensed_pwr /  dirac_0),
	xerr = Array{Float64,1}[bin_mids_T-lftcuts, rtcuts-bin_mids_T],  
	yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), tib_unlensed_pwr /  dirac_0), map(x-> quantile(x,0.975)-median(x), tib_unlensed_pwr /  dirac_0)],
	fmt="*b",
	label = L"95\% posterior region for $l^2 |T_l|^2/ \delta_0$"
)
plot(collect(bin_mids_T), zero(bin_mids_T), ":k")
xlabel("wavenumber")
legend()
axis("tight")
ticklabel_format(style="sci", axis="y", scilimits=(0,0))
savefig(joinpath(savepath, "figure8b.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)








"""
Cross correlation for phi and T
------------------------------------------
"""
rtcuts  = collect(bin_mids_P +  step(bin_mids_P) / 2)  
lftcuts = collect(bin_mids_P -  step(bin_mids_P) / 2)  
lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
errorbar(
	bin_mids_P,
	map(median,  phb_cov),
	xerr = Array{Float64,1}[bin_mids_P-lftcuts, rtcuts-bin_mids_P],  
	yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), phb_cov), 
							map(x-> quantile(x,0.975)-median(x), phb_cov)],
	fmt="*b",
	label = L"empirical cross correlation with $\phi_l$"
)
annotate(L"corr between $\phi_l$ and $P(\phi_l|data)$",
         xy=(bin_mids_P[20], map(median,  phb_cov)[20]+0.02), xycoords="data",
         xytext=(-10, 80), textcoords="offset points", fontsize=16,
         arrowprops={:arrowstyle=>"->"})
rtcuts  = collect(bin_mids_T +  step(bin_mids_T) / 2)  
lftcuts = collect(bin_mids_T -  step(bin_mids_T) / 2) 
lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero 
errorbar(
	bin_mids_T,
	map(median,  tib_unlensed_cov),
	xerr = Array{Float64,1}[bin_mids_T-lftcuts, rtcuts-bin_mids_T],  
	yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), tib_unlensed_cov), 
							map(x-> quantile(x,0.975)-median(x), tib_unlensed_cov)],
	fmt="*b",
	label = L"empirical cross correlation with $T_l$"
)
annotate(L"corr between $T_l$ and $P(T_l|data)$",
         xy=(bin_mids_T[30], map(median,  tib_unlensed_cov)[30]-0.02), xycoords="data",
         xytext=(-170, -60), textcoords="offset points", fontsize=16,
         arrowprops={:arrowstyle=>"->"})
#plot(collect(bin_mids_P), zero(bin_mids_P), ":k")
xlabel("wavenumber")
#legend()
axis("tight")
ylim(-0.05,1)
savefig(joinpath(savepath, "figure9a.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)





"""
trace plots
------------------------------------------"""
simdir_trc   =  "/Users/ethananderes/Dropbox/BayesLense/simulations/scriptTrack_2657077506/" # contains the simulation
trace_trc   = map(include_string, readcsv("$(simdir_trc)/trace.csv"))

# ----This is the degree reference for trace_trc[3, :]
# [(180/pi) * parlr.grd.x[400,300], (180/pi) * parlr.grd.y[400,300]] #<--- [9.9,13.2]
# ----This is the degree reference for trace_trc[4, :] 
# [(180/pi) * parlr.grd.x[50,50], (180/pi) *  parlr.grd.y[50,50]] #<---------- [1.6, 1.6]
plot(vcat(0.0, real(trace_trc[3,2:5001])...),     color = "blue", alpha = 0.5,  linewidth=1.5, linestyle="-", label = L"\phi(x),\, x = (9.9^o, 13.2^o)")
plot(zeros(size(trace_trc,2)) + trace_trc[3,1],  color = "blue", alpha = 1.0,  linewidth=2.5, linestyle="--")
plot(vcat(0.0, real(trace_trc[4,2:5001])...),     color = "green", alpha = 0.5,  linewidth=1.5, linestyle="-",  label = L"\phi(x),\, x = (1.6^o, 1.6^o)")
plot(zeros(size(trace_trc,2)) + trace_trc[4,1],  color = "green", alpha = 1.0,  linewidth=2.5, linestyle="--")
xlabel("Gibbs iteration")
ylabel(L"Gibbs chain values of $\phi(x)$")
ticklabel_format(style="sci", axis="y", scilimits=(0,0))
title("Assessing Gibbs chain convergence")
legend(loc="best", fontsize = 12)
axis("tight")
savefig(joinpath(savepath, "figure10a.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)




# ----This is the degree reference for trace_trc[6, :]
# k = [parlr.grd.k1[4, 7],  parlr.grd.k2[4, 7]] #<----  [126.56,63.28]
# wavenumber = norm(k) #<--- 141.50
plot(vcat(0.0, real(trace_trc[6,2:5001])...),     color = "blue", alpha = 0.5,  linewidth=1.5, linestyle="-", label = L"real(\phi_l),\,l = (126.56,63.28)")
plot(zeros(size(trace_trc,2)) + real(trace_trc[6,1]), color = "blue", alpha = 1.0,  linewidth=2.5, linestyle="--")
plot(vcat(0.0, imag(trace_trc[6,2:5001])...),  color = "green", alpha = 0.5,  linewidth=1.5, linestyle="-", label = L"imag(\phi_l),\,l = (126.56,63.28)")
plot(zeros(size(trace_trc,2)) + imag(trace_trc[6,1]),  color = "green", alpha = 1.0,  linewidth=2.5, linestyle="--")
xlabel("Gibbs iteration")
ylabel(L"Gibbs chain values of $\phi_l$")
title("Assessing Gibbs chain convergence")
ticklabel_format(style="sci", axis="y", scilimits=(0,0))
legend(loc="best", fontsize = 12)
axis("tight")
savefig(joinpath(savepath, "figure10b.pdf"), dpi=300, bbox_inches="tight",  pad_inches=0.1)






