#=
	include("scripts/makeFigsParallel.jl")
=#

 
# run this for the paper
krang = 551:100:2501 # range of samples we are looking at
jobs = 15
savebool = true # set to true if you want the images saved
simdir     =  "scriptParallel_2657077506"
savefilepath = "/Users/ethananderes/Dropbox/BayesLense/paper/fromParallel"



#=
# run this for experimentation
krang = 1:10:200 # range of samples we are looking at
jobs = 15
savebool = true # set to true if you want the images saved
simdir     =  "scriptParallel_1743177682"
savefilepath = "/Users/ethananderes/Desktop"
=#

using PyPlot
push!(LOAD_PATH, pwd()*"/src")
using Interp
require("cmb.jl")
require("fft.jl")
require("funcs.jl")


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
	"src"
	);
dirac_0 = 1/parlr.grd.deltk^d 
elp = parlr.ell[10:int(maskupP)] 
elt = parlr.ell[10:1000] 
cpl = parlr.CPell2d[10:int(maskupP)]
ctl = parlr.CTell2dLen[10:1000]
r2 = parlr.grd.r.^2
bin_mids_P = (parlr.grd.deltk*1.5):(parlr.grd.deltk):maskupP
bin_mids_T = (parlr.grd.deltk*1.5):(parlr.grd.deltk):1000



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



######################################
#
# read in the data
#
#######################################
# --- initalize the containers
# truths
phix = readcsv("simulations/$simdir/phix.csv")
tildex = readcsv("simulations/$simdir/tildetx_lr.csv")
phik =   fft2(phix, parlr)
tildek = fft2(tildex, parlr)
ytx  = readcsv("simulations/$simdir/ytx.csv")
qex = readcsv("simulations/$simdir/qex_lr.csv")
x = readcsv("simulations/$simdir/x.csv")


# cross correlation
phik_cov_smpl = Array{Float64,1}[]
tildek_cov_smpl = Array{Float64,1}[]

# spectral coverage
phik_pwr_smpl = Array{Float64,1}[]
tildek_pwr_smpl = Array{Float64,1}[]

# 1-d slices
propslice = 0.75 # btwn 0 and 1
phix_slice_samples = Array{Float64,1}[]
tildex_slice_samples = Array{Float64,1}[]

# average 
phix_sum = zero(phix)
tildetx_lr_sum = zero(phix)



cntr = 0
for r in 1:jobs, k in krang
	if isfile("simulations/$simdir/job$r/phix_curr_$k.csv") 
	
		phix_curr = readcsv("simulations/$simdir/job$r/phix_curr_$k.csv")
		phik_curr = fft2(phix_curr, parlr)
		tildetx_curr = readcsv("simulations/$simdir/job$r/tildetx_lr_curr_$k.csv")
		tildetk_curr = fft2(tildetx_curr, parlr)
	
		# spectral coverage
		push!(phik_pwr_smpl,   binpower(r2    .* phik_curr, parlr.grd.r, bin_mids_P))
		push!(tildek_pwr_smpl, binpower(√(r2) .* tildetk_curr, parlr.grd.r, bin_mids_T))
	
		# cross correlation
		pnxt   = binave(  r2 .* phik_curr .* conj(phik), parlr.grd.r, bin_mids_P)
		pnxt ./= binpower(sqrt(r2) .* phik_curr, parlr.grd.r, bin_mids_P) |> sqrt
		pnxt ./= binpower(sqrt(r2) .* phik, parlr.grd.r, bin_mids_P) |> sqrt
		tnxt   = binave(  r2 .* tildetk_curr .* conj(tildek), parlr.grd.r, bin_mids_T)
		tnxt ./= binpower(sqrt(r2) .* tildetk_curr, parlr.grd.r, bin_mids_T) |> sqrt
		tnxt ./= binpower(sqrt(r2) .* tildek, parlr.grd.r, bin_mids_T) |> sqrt
		push!(phik_cov_smpl, real(pnxt))
		push!(tildek_cov_smpl, real(tnxt))
	
		# 1-d slices
		push!(phix_slice_samples, phix_curr[int(end*propslice),:][:])
		push!(tildex_slice_samples, tildetx_curr[int(end*propslice),:][:])
	
		# averages
		phix_sum += phix_curr
		tildetx_lr_sum += tildetx_curr

	cntr += 1
	end
end

# 1-d slices
phix_slice_samples   = hcat(phix_slice_samples...)
tildex_slice_samples = hcat(tildex_slice_samples...)
phix_true_slice      = phix[int(end*propslice),:][:]
tildex_true_slice    = tildex[int(end*propslice),:][:]
phix_sum_slice  = phix_sum[int(end*propslice),:][:]
tildetx_lr_sum_slice = tildetx_lr_sum[int(end*propslice),:][:]
qex_slice = qex[int(end*propslice),:][:]
x_slice = x[int(end*propslice),:][:]
varx_slice = readcsv("simulations/$simdir/maskvarx.csv")[int(end*propslice),:][:]
isempty(x_slice[varx_slice.==Inf]) || (maskmin=minimum(x_slice[varx_slice.==Inf]); maskmax = maximum(x_slice[varx_slice.==Inf]))


# cross correlation
# squish to matrix
phik_cov_smpl   = hcat(phik_cov_smpl...)
tildek_cov_smpl = hcat(tildek_cov_smpl...)
# then split by the rows
phb_cov = Array{Float64,1}[vec(phik_cov_smpl[k,:])   for k=1:size(phik_cov_smpl,   1)]
tib_cov = Array{Float64,1}[vec(tildek_cov_smpl[k,:]) for k=1:size(tildek_cov_smpl, 1)]
# now phb[2] shoudl be the samples over bin bin_mids[2]

# spectral covarge
# squish to matrix
phik_pwr_smpl   = hcat(phik_pwr_smpl...)
tildek_pwr_smpl = hcat(tildek_pwr_smpl...)
# then split by the rows
phb_pwr = Array{Float64,1}[phik_pwr_smpl[k,:][:]   for k=1:size(phik_pwr_smpl,   1)]
tib_pwr = Array{Float64,1}[tildek_pwr_smpl[k,:][:] for k=1:size(tildek_pwr_smpl, 1)]
# now phb[2] shoudl be the samples over bin bin_mids[2]

# here is the truth
phb_truth = binpower(r2 .* phik, parlr.grd.r, bin_mids_P)
tib_truth = binpower(sqrt(r2) .* tildek, parlr.grd.r, bin_mids_T)
	




######################################
#
# make the plots
#
#######################################


# ------  acceptance rate plots
accptrec = readcsv("simulations/$simdir/job1/acceptclk.csv")		
# plots the acceptence rate in a sliding window of 10, after removing the gradient records
sliding_ave = Float64[]
k = 1
wdsize = 50
accptrec_notwos = accptrec[accptrec .< 2]
while true
	(k+wdsize > length(accptrec_notwos)) && break
	push!(sliding_ave, mean(accptrec_notwos[k:k+wdsize]))
	k += 1
end
fg = figure()
plot(sliding_ave)
plot(zero(sliding_ave))
plot(zero(sliding_ave)+1)
ylabel("acceptance rate")
xlabel("iteration")
if savebool; savefig(joinpath(savefilepath, "acceptRate.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)





# ---------- spectral coverage for phi
fg = figure()
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
axis("tight")
if savebool; savefig(joinpath(savefilepath, "specP.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)



# ---------- spectral coverage for T
fg = figure()
plot(elt,  elt.^2 .* ctl, "-k", label = L"$l^2C_l^{ \tilde T\tilde T}$")
plot(bin_mids_T, tib_truth / dirac_0 , "or", label =  L"$l^2 |\widetilde T_l|^2/ \delta_0$")
rtcuts  = collect(bin_mids_T +  step(bin_mids_T) / 2)  
lftcuts = collect(bin_mids_T -  step(bin_mids_T) / 2)  
lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
errorbar(
	bin_mids_T,
	map(median,  tib_pwr /  dirac_0),
	xerr = Array{Float64,1}[bin_mids_T-lftcuts, rtcuts-bin_mids_T],  
	yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), tib_pwr /  dirac_0), map(x-> quantile(x,0.975)-median(x), tib_pwr /  dirac_0)],
	fmt="*b",
	label = L"95\% posterior region for $l^2 |\widetilde T_l|^2/ \delta_0$"
)
plot(collect(bin_mids_T), zero(bin_mids_T), ":k")
xlabel("wavenumber")
legend()
axis("tight")
if savebool; savefig(joinpath(savefilepath, "specT.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)


#=

# -------- corrlation btwn fourier modes
md = 10
scatter(tib_pwr[md], tib_pwr[md+1])
xlabel("power at wave number $(bin_mids_T[md])")
ylabel("power at wave number $(bin_mids_T[md+1])")

=#



# ------- cooling schedule
fg = plt.figure()
lcool = repstretch(2.0, 1.2maskupC, 75, 1000)
λcool = Float64[]
d2x   = parlr.grd.deltx * parlr.grd.deltx
d2k   = parlr.grd.deltk * parlr.grd.deltk
delt0 = 1 / d2k
barNx = 0.99 * parlr.nugget_at_each_pixel
barNk = barNx * d2x * delt0 # var in fourier
for l in lcool
	λbarNk = (l > 8000.0) ? barNk : max(barNk, delt0 * parlr.CTell2d[int(l)]) 
	push!(λcool, λbarNk / barNk)
end
semilogy(λcool)
xlabel(L"message passing iteration $k$", fontsize=22)
ylabel(L"$\lambda^k$", fontsize=28)
axis("tight")
if savebool; savefig(joinpath(savefilepath, "cooling.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)







# -------- cross correlation for phi
fg = plt.figure()
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
plot(collect(bin_mids_P), zero(bin_mids_P), ":k")
xlabel("wavenumber")
legend()
axis("tight")
if savebool; savefig(joinpath(savefilepath, "corrP.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)




# -------- cross correlation for T
fg = plt.figure()
rtcuts  = collect(bin_mids_T +  step(bin_mids_T) / 2)  
lftcuts = collect(bin_mids_T -  step(bin_mids_T) / 2)  
lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
errorbar(
	bin_mids_T,
	map(median,  tib_cov),
	xerr = Array{Float64,1}[bin_mids_T-lftcuts, rtcuts-bin_mids_T],  
	yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), tib_cov), 
							map(x-> quantile(x,0.975)-median(x), tib_cov)],
	fmt="*b",
	label = L"empirical cross correlation with $\widetilde T_l$"
)
plot(collect(bin_mids_T), zero(bin_mids_T), ":k")
xlabel("wavenumber")
legend(loc= "lower right")
axis("tight")
if savebool; savefig(joinpath(savefilepath, "corrT.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)






# -------- true phix
fg = plt.figure()
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
if savebool; savefig(joinpath(savefilepath, "phix.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)





# ------- quadratic estimate
fg = plt.figure()
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
if savebool; savefig(joinpath(savefilepath, "qex.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)




# ------ posterior mean for phix
fg = plt.figure()
imshow(
  phix_sum/cntr, 
  interpolation = "nearest", 
  vmin=minimum(phix),
  vmax=maximum(phix), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
plt.xlabel("degrees")
plt.ylabel("degrees")
if savebool; savefig(joinpath(savefilepath, "phix_est.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)





# ------   truth tildex
fg = plt.figure()
imshow(
  tildex, 
  interpolation = "nearest", 
  vmin = minimum(tildex),
  vmax = maximum(tildex), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
plt.xlabel("degrees")
plt.ylabel("degrees")
if savebool; savefig(joinpath(savefilepath, "tildex.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)




# --------- poster mean tildex
fg = plt.figure()
imshow(
  tildetx_lr_sum/cntr, 
  interpolation = "nearest", 
  vmin=minimum(tildex),
  vmax=maximum(tildex), 
  origin="lower", 
  extent=(180/pi)*[minimum(x), maximum(x),minimum(x), maximum(x)]
) 
plt.xlabel("degrees")
plt.ylabel("degrees")
if savebool; savefig(joinpath(savefilepath, "tildex_est.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)





# -------- slices of phix
fg = figure()
plot(x_slice, phix_slice_samples[:,1], color="blue", alpha=0.15, label=L"posterior samples of $\phi(x)$")
plot(x_slice, phix_slice_samples[:,2:end], color="blue", alpha=0.15)
plot(x_slice, phix_true_slice, color = "red",  linewidth=1.5, linestyle="-", label=L"simluation true $\phi(x)$")
annotate("mask",
         xy=(maskmin, 0), xycoords="data",
         xytext=(-50, -50), textcoords="offset points", fontsize=16,
         arrowprops={:arrowstyle=>"->"})
xlabel("radians")
axis("tight")
axvspan(maskmin, maskmax,  facecolor="0.5", alpha=0.3, label="mask region")
legend(loc="upper left")
if savebool; savefig(joinpath(savefilepath, "phix_slice.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)




# ------- slice of post mean of phix
fg = figure()
plot(x_slice, phix_true_slice, color = "red",  linewidth=1.5, linestyle="-", label=L"simluation true $\phi(x)$")
plot(x_slice, qex_slice, color = "black",  linewidth=1.5, linestyle=":", label=L"quadratic est")
plot(x_slice,  phix_sum_slice/cntr, color = "blue", linewidth=1.5, linestyle="--",  alpha=0.8, label=L"posterior mean")
annotate("mask",
         xy=(maskmin, 0), xycoords="data",
         xytext=(-70, 10), textcoords="offset points", fontsize=16,
         arrowprops={:arrowstyle=>"->"})
xlabel("radians")
axis("tight")
axvspan(maskmin, maskmax,  facecolor="0.5", alpha=0.3, label="mask region")
legend(loc="upper right")
if savebool; savefig(joinpath(savefilepath, "phix_ave_slice.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)




# ----- slice of tildex
fg = figure()
lper = 0.2
rper = 0.5
zoom = round(lper*length(x_slice)):round(rper*length(x_slice))
plot(x_slice[zoom], tildex_slice_samples[zoom,1], color="blue", alpha=0.15, label=L"posterior samples $\tilde T(x)$")
plot(x_slice[zoom], tildex_slice_samples[zoom,2:end], color="blue", alpha=0.15)
plot(x_slice[zoom], tildex_true_slice[zoom], color = "red",  linewidth=1.4, linestyle="-", label=L"simulation true $\tilde T(x)$")
axis("tight")
axvspan(maskmin, maskmax,  facecolor="0.5", alpha=0.3, label="mask region")
annotate("mask", xy=(maskmin, 0), xycoords="data", xytext=(-50, -50), textcoords="offset points", fontsize=16, arrowprops={:arrowstyle=>"->"})
xlabel("radians")
axis("tight")
legend(loc="upper left")
if savebool; savefig(joinpath(savefilepath, "tildex_ave_slice.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)


# ---- data 
fg = figure()
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
if savebool; savefig(joinpath(savefilepath, "data.pdf"), dpi=200, bbox_inches="tight") end 
close(fg)



