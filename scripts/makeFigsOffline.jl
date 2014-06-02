#=
include("scripts/makeFigsOffline.jl")
=#
const simdir    =  "scriptNew_MaskDv2_10000" 
const specc 	  = false  # plot the spectral coverage
const pcorr 	  = false  # plot the empirical cross correlation
const onedslice  = false  # plot the 1-d slices of phi
const acc 	     = false # take a look at the acceptance rate
const imagsli    = false # look at the images one by one
const aveim      = true  # point-wise average.
const mvie 	     = false # <---- needs work
const krang = 1:1:5000   # range of samples we are looking at

# --- copy these are from the runfile
const scriptname = "scriptNew"
const percentNyqForC = 0.5 # used for T l_max
const numofparsForP  = 1500  # used for P l_max
const hrfactor = 2.0
const pixel_size_arcmin = 2.0
const n = 2.0^9
const beamFWHM = 0.0
const nugget_at_each_pixel = (3.0)^2
begin  #< ---- dependent run parameters
	local deltx =  pixel_size_arcmin * pi / (180 * 60) #rads
	local period = deltx * n # side length in rads
	local deltk =  2 * pi / period
	local nyq = (2 * pi) / (2 * deltx)
	const maskupP  = sqrt(deltk^2 * numofparsForP / pi)  #l_max for for phi
	const maskupC  = min(9000.0, percentNyqForC * (2 * pi) / (2 * pixel_size_arcmin * pi / (180*60))) #l_max for for phi
	println("muK_per_arcmin = $(sqrt(nugget_at_each_pixel * (pixel_size_arcmin^2)))") # muK per arcmin
	println("maskupP = $maskupP") # muK per arcmin
	println("maskupC = $maskupC") # muK per arcmin
end
const scale_grad =  1.0e-3
const scale_hmc  =  1.0e-3


# ---- modules etc
using PyCall 
@pyimport matplotlib.pyplot as plt
push!(LOAD_PATH, "src")
using Interp
require("cmb.jl")
require("fft.jl")


#---------------------------------------------------
# plot the spectral coverage
#-----------------------------------------
if specc
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
	elt = parlr.ell[10:int(maskupC)] 
	cpl = parlr.CPell2d[10:int(maskupP)]
	ctl = parlr.CTell2d[10:int(maskupC)]
	r2 = parlr.grd.r.^2
	bin_mids_P = (parlr.grd.deltk*2):(parlr.grd.deltk):maskupP
	bin_mids_T = (parlr.grd.deltk*2):(parlr.grd.deltk):maskupC
	
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
	
	
	phik_pwr_smpl = Array{Float64,1}[]
	tildek_pwr_smpl = Array{Float64,1}[]
	cntr = 0
	for k in krang
		if isfile("simulations/$simdir/phix_curr_$k.csv") & isfile("simulations/$simdir/tildetx_lr_curr_$k.csv")
			phik_curr    = fft2(readcsv("simulations/$simdir/phix_curr_$k.csv"), parlr) 
			tildetk_curr = fft2(readcsv("simulations/$simdir/tildetx_lr_curr_$k.csv"), parlr)
			push!(phik_pwr_smpl,   binpower(r2 .* phik_curr, parlr.grd.r, bin_mids_P))
			push!(tildek_pwr_smpl, binpower(sqrt(r2) .* tildetk_curr, parlr.grd.r, bin_mids_T))
			cntr += 1
		end
	end
	# squish to matrix
	phik_pwr_smpl   = hcat(phik_pwr_smpl...)
	tildek_pwr_smpl = hcat(tildek_pwr_smpl...)
	# then split by the rows
	phb = Array{Float64,1}[phik_pwr_smpl[k,:][:]   for k=1:size(phik_pwr_smpl,   1)]
	tib = Array{Float64,1}[tildek_pwr_smpl[k,:][:] for k=1:size(tildek_pwr_smpl, 1)]
	# now phb[2] shoudl be the samples over bin bin_mids[2]
	
	# here is the truth
	phik =   fft2(readcsv("simulations/$simdir/phix.csv"), parlr)
	tildek = fft2(readcsv("simulations/$simdir/tildetx_lr.csv"), parlr)
	
	phb_truth = binpower(r2 .* phik, parlr.grd.r, bin_mids_P)
	tib_truth = binpower(sqrt(r2) .* tildek, parlr.grd.r, bin_mids_T)
	
	plt.figure(figsize=(15,5))
	plt.plot(elp, dirac_0 .* elp.^4 .* cpl, "-k")
	plt.plot(bin_mids_P, phb_truth, "or", label = "truth")
	rtcuts  = collect(bin_mids_P +  step(bin_mids_P) / 2)  
	lftcuts = collect(bin_mids_P -  step(bin_mids_P) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	plt.errorbar(
		bin_mids_P,
		map(median,  phb),
		xerr = Array{Float64,1}[bin_mids_P-lftcuts, rtcuts-bin_mids_P],  
		yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), phb), map(x-> quantile(x,0.975)-median(x), phb)],
		fmt="*b",
		label = "posterior"
	)
	plt.legend()
	plt.show()


	plt.figure(figsize=(15,5))
	plt.plot(elt, dirac_0 .* elt.^2 .* ctl, "-k")
	plt.plot(bin_mids_T, tib_truth, "or", label = "truth")
	rtcuts  = collect(bin_mids_T +  step(bin_mids_T) / 2)  
	lftcuts = collect(bin_mids_T -  step(bin_mids_T) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	plt.errorbar(
		bin_mids_T,
		map(median,  tib),
		xerr = Array{Float64,1}[bin_mids_T-lftcuts, rtcuts-bin_mids_T],  
		yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), tib), map(x-> quantile(x,0.975)-median(x), tib)],
		fmt="*b",
		label = "posterior"
	)
	plt.legend()
	plt.show()
end 


#---------------------------------------------------
# plot cross correlation
#-----------------------------------------
if pcorr
	d = 2
	bin_mids = 20:20:900
	
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
	elt = parlr.ell[10:int(maskupC)] 
	cpl = parlr.CPell2d[10:int(maskupP)]
	ctl = parlr.CTell2d[10:int(maskupC)]
	r2 = parlr.grd.r.^2
	bin_mids_P = (parlr.grd.deltk*2):(parlr.grd.deltk):maskupP
	bin_mids_T = (parlr.grd.deltk*2):(parlr.grd.deltk):maskupC

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
	
	phik =   fft2(readcsv("simulations/$simdir/phix.csv"), parlr)
	tildek = fft2(readcsv("simulations/$simdir/tildetx_lr.csv"), parlr)
	
	phik_cov_smpl = Array{Float64,1}[]
	tildek_cov_smpl = Array{Float64,1}[]
	cntr = 0
	for k in krang
		if isfile("simulations/$simdir/phix_curr_$k.csv") && isfile("simulations/$simdir/tildetx_lr_curr_$k.csv")
			phik_curr    = fft2(readcsv("simulations/$simdir/phix_curr_$k.csv"), parlr) 
			tildetk_curr = fft2(readcsv("simulations/$simdir/tildetx_lr_curr_$k.csv"), parlr)
			pnxt   = binave(  r2 .* phik_curr .* conj(phik), parlr.grd.r, bin_mids_P)
			pnxt ./= binpower(sqrt(r2) .* phik_curr, parlr.grd.r, bin_mids_P) |> sqrt
			pnxt ./= binpower(sqrt(r2) .* phik, parlr.grd.r, bin_mids_P) |> sqrt
			tnxt   = binave(  r2 .* tildetk_curr .* conj(tildek), parlr.grd.r, bin_mids_T)
			tnxt ./= binpower(sqrt(r2) .* tildetk_curr, parlr.grd.r, bin_mids_T) |> sqrt
			tnxt ./= binpower(sqrt(r2) .* tildek, parlr.grd.r, bin_mids_T) |> sqrt
			push!(phik_cov_smpl, real(pnxt))
			push!(tildek_cov_smpl, real(tnxt))
			cntr += 1
		end
	end
	# squish to matrix
	phik_cov_smpl   = hcat(phik_cov_smpl...)
	tildek_cov_smpl = hcat(tildek_cov_smpl...)
	# then split by the rows
	phb = Array{Float64,1}[vec(phik_cov_smpl[k,:])   for k=1:size(phik_cov_smpl,   1)]
	tib = Array{Float64,1}[vec(tildek_cov_smpl[k,:]) for k=1:size(tildek_cov_smpl, 1)]
	# now phb[2] shoudl be the samples over bin bin_mids[2]
	
	# --------- for phi
	plt.figure(figsize=(15,5))
	rtcuts  = collect(bin_mids_P +  step(bin_mids_P) / 2)  
	lftcuts = collect(bin_mids_P -  step(bin_mids_P) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	plt.errorbar(
		bin_mids_P,
		map(median,  phb),
		xerr = Array{Float64,1}[bin_mids_P-lftcuts, rtcuts-bin_mids_P],  
		yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), phb), 
								map(x-> quantile(x,0.975)-median(x), phb)],
		fmt="*b",
		label = "empirical correlation with truth"
	)
	plt.plot(collect(bin_mids_P), zero(bin_mids_P), ":k")
	plt.legend()
	plt.show()
	#  ------------ for T
	plt.figure(figsize=(15,5))
	rtcuts  = collect(bin_mids_T +  step(bin_mids_T) / 2)  
	lftcuts = collect(bin_mids_T -  step(bin_mids_T) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	plt.errorbar(
		bin_mids_T,
		map(median,  tib),
		xerr = Array{Float64,1}[bin_mids_T-lftcuts, rtcuts-bin_mids_T],  
		yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), tib), 
								map(x-> quantile(x,0.975)-median(x), tib)],
		fmt="*b",
		label = "empirical correlation with truth"
	)
	plt.plot(collect(bin_mids_T), zero(bin_mids_T), ":k")
	plt.legend()
	plt.show()
end 


#---------------------------------------------------
# plot the 1-d slices of phi
#-----------------------------------------
if onedslice
	propslice = 0.8 # btwn 0 and 1
	phix_slice_samples = Array{Float64,1}[]
	tildex_slice_samples = Array{Float64,1}[]
	cntr = 0
	for k in krang
		if isfile("simulations/$simdir/phix_curr_$k.csv") & isfile("simulations/$simdir/tildetx_lr_curr_$k.csv")
			phix_curr = readcsv("simulations/$simdir/phix_curr_$k.csv")
			tildetx_curr = readcsv("simulations/$simdir/tildetx_lr_curr_$k.csv")
			push!(phix_slice_samples, phix_curr[int(end*propslice),:][:])
			push!(tildex_slice_samples, tildetx_curr[int(end*propslice),:][:])
			cntr += 1
		end
	end
	phix_slice_samples = hcat(phix_slice_samples...)
	tildex_slice_samples = hcat(tildex_slice_samples...)
	
	# files I can upload directly
	phix = readcsv("simulations/$simdir/phix.csv")[int(end*propslice),:][:]
	tildex = readcsv("simulations/$simdir/tildetx_lr.csv")[int(end*propslice),:][:]
	qex = readcsv("simulations/$simdir/qex_lr.csv")[int(end*propslice),:][:]
	x = readcsv("simulations/$simdir/x.csv")[int(end*propslice),:][:]
	varx = readcsv("simulations/$simdir/maskvarx.csv")[int(end*propslice),:][:]
	isempty(x[varx.==Inf]) || (maskmin=minimum(x[varx.==Inf]); maskmax = maximum(x[varx.==Inf]))
	
	# plot phix
	plt.plot(x, phix_slice_samples[:,1], color="blue", alpha=0.4, label="posterior samples")
	plt.plot(x, phix_slice_samples[:,2:end], color="blue", alpha=0.4)
	plt.plot(x, phix, color = "red",  linewidth=2.5, linestyle="-", label="truth")
	plt.plot(x, qex, color = "green",  linewidth=2.5, linestyle="--", label="quadratic estimate")
	plt.plot(x,  mean(phix_slice_samples,2), color = "black", linewidth=2.5, linestyle="--",  alpha=0.8, label="posterior mean")
	plt.plot(x, zero(phix), color = "black",  linestyle=":")
	plt.legend(loc="upper right")
	isempty(x[varx.==Inf]) || plt.annotate("mask",
	         xy=(maskmin, 0), xycoords="data",
	         xytext=(-50, -50), textcoords="offset points", fontsize=16,
	         arrowprops={:arrowstyle=>"->"})
	plt.title("Lensing potential posterior samples")
	plt.xlabel("radians")
	isempty(x[varx.==Inf]) || plt.axvspan(maskmin, maskmax,  facecolor="0.5", alpha=0.3, label="mask region")
	plt.show()
	
	
	#plot tildex
	plt.plot(x, tildex_slice_samples[:,1], color="blue", alpha=0.4, label="posterior samples")
	plt.plot(x, tildex_slice_samples[:,2:end], color="blue", alpha=0.4)
	plt.plot(x, tildex, color = "red",  linewidth=2.5, linestyle="-", label="truth")
	plt.plot(x,  mean(tildex_slice_samples,2), color = "black", linewidth=2.5, linestyle="--",  alpha=0.8, label="posterior mean")
	plt.plot(x, zero(tildex), color = "black",  linestyle=":")
	plt.legend(loc="upper right")
	isempty(x[varx.==Inf]) || plt.annotate("mask",
	         xy=(maskmin, 0), xycoords="data",
	         xytext=(-50, -50), textcoords="offset points", fontsize=16,
	         arrowprops={:arrowstyle=>"->"})
	plt.title("Lensed CMB posterior samples")
	plt.xlabel("radians")
	isempty(x[varx.==Inf]) || plt.axvspan(maskmin, maskmax,  facecolor="0.5", alpha=0.3, label="mask region")
	plt.show()
end 


#---------------------------------------------------
# take a look at the accpetence rate
#-----------------------------------------
if acc 
	accptrec = readcsv("simulations/$simdir/acceptclk.csv")
	
	# plots the record
	# plt.plot(accptrec)
	# plt.show()
	
	# plots the acceptence rate in a sliding window of 10, after removing the gradient records
	sliding_ave = Float64[]
	k = 1
	wdsize = 50
	accptrec_notwos = accptrec[accptrec .< 2]
	while true
		if k+wdsize > length(accptrec_notwos); break; end 
		push!(sliding_ave, mean(accptrec_notwos[k:k+wdsize]))
		k += 1
	end
	plt.plot(sliding_ave)
	plt.show()
end #end if


#---------------------------------
# look at the images one by one
#--------------------------------
if  imagsli 
	phix = readcsv("simulations/$simdir/phix.csv")
	tildex = readcsv("simulations/$simdir/tildetx_lr.csv")
	ytx  = readcsv("simulations/$simdir/ytx.csv")
	for k in krang
		if isfile("simulations/$simdir/phix_curr_$k.csv") & isfile("simulations/$simdir/tildetx_lr_curr_$k.csv")
			phix_curr = readcsv("simulations/$simdir/phix_curr_$k.csv")
			tildetx_curr = readcsv("simulations/$simdir/tildetx_lr_curr_$k.csv")
			plt.figure(figsize=(10,10))
			plt.subplot(2,2,3)
			plt.imshow(phix, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
			plt.xlabel("true lensing potential")
			plt.subplot(2,2,1)
			plt.imshow(phix_curr, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
			plt.xlabel("estimated lensing potential")
			plt.subplot(2,2,2)
			plt.imshow(tildetx_curr, interpolation = "none", vmin=minimum(ytx),vmax=maximum(ytx)) 
			plt.xlabel("estimated unlensing CMB")
			plt.subplot(2,2,4)
			plt.imshow(ytx, interpolation = "none", vmin=minimum(ytx),vmax=maximum(ytx)) 
			plt.xlabel("data")
			plt.show()
		end
	end
end # end the if

#--------------------
# pointwise average
#---------------
if aveim
	phix = readcsv("simulations/$simdir/phix.csv")
	tildex = readcsv("simulations/$simdir/tildetx_lr.csv")
	ytx  = readcsv("simulations/$simdir/ytx.csv")
	qex = readcsv("simulations/$simdir/qex_lr.csv")
	phix_sum = zero(phix)
	tildetx_lr_sum = zero(phix)
	cntr = 0
	for k in krang
		if isfile("simulations/$simdir/phix_curr_$k.csv") & isfile("simulations/$simdir/tildetx_lr_curr_$k.csv")
			phix_curr = readcsv("simulations/$simdir/phix_curr_$k.csv")
			tildetx_lr_curr = readcsv("simulations/$simdir/tildetx_lr_curr_$k.csv")
			phix_sum += phix_curr
			tildetx_lr_sum += tildetx_lr_curr
			cntr += 1
		end
	end
	fig = plt.figure(figsize=(15,6))
	plt.subplot(1,3,1)
	plt.imshow(phix, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
	plt.xlabel("true lensing potential")
	plt.subplot(1,3,2)
	plt.imshow(phix_sum/cntr, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
	plt.xlabel("estimated lensing potential")
	plt.subplot(1,3,3)
	plt.imshow(qex, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
	plt.xlabel("quadratic estimate")
	#=plt.imshow(tildetx_lr_sum/cntr, interpolation = "none", vmin=minimum(ytx),vmax=maximum(ytx)) 
	plt.xlabel("estimated unlensing CMB")
	plt.subplot(2,2,4)
	plt.imshow(ytx, interpolation = "none", vmin=minimum(ytx),vmax=maximum(ytx)) 
	plt.xlabel("data")=#
	plt.show()
end

#---------------------------------
#  make movies and average estimate
# -----------------------------------------
if mvie 
tm = "/tmp/trashthismovie"
dr = tm * "/mov"
isdir(dr) || (mkdir(tm) ; mkdir(dr))

function saveMovieFram(dr, num, phix, phix_curr, ytx, tildetx_lr_curr)
	# save to dr
	fig = plt.figure(figsize=(10,10))
	plt.subplot(2,2,3)
	plt.imshow(phix, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
	plt.xlabel("true lensing potential")
	plt.subplot(2,2,1)
	plt.imshow(phix_curr, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
	plt.xlabel("estimated lensing potential")
	plt.subplot(2,2,2)
	plt.imshow(tildetx_lr_curr, interpolation = "none", vmin=minimum(ytx),vmax=maximum(ytx)) 
	plt.xlabel("estimated unlensing CMB")
	plt.subplot(2,2,4)
	plt.imshow(ytx, interpolation = "none", vmin=minimum(ytx),vmax=maximum(ytx)) 
	plt.xlabel("data")
	plt.savefig(dr * "/" * string(num) * ".png")
	plt.close(fig)
end
phix = readcsv("simulations/$simdir/phix.csv")
ytx  = readcsv("simulations/$simdir/ytx.csv")
phix_sum = zero(phix)
tildetx_lr_sum = zero(phix)
cntr = 0
for k in krang
	if isfile("simulations/$simdir/phix_curr_$k.csv") & isfile("simulations/$simdir/tildetx_lr_curr_$k.csv")
		phix_curr = readcsv("simulations/$simdir/phix_curr_$k.csv")
		tildetx_lr_curr = readcsv("simulations/$simdir/tildetx_lr_curr_$k.csv")
		saveMovieFram(dr, cntr, phix, phix_curr, ytx, tildetx_lr_curr)
		phix_sum += phix_curr
		tildetx_lr_sum += tildetx_lr_curr
		cntr += 1
	end
end
saveMovieFram(dr, cntr, phix, phix_sum/cntr, ytx, tildetx_lr_sum/cntr)
run(`/Applications/ffmpeg -y -i $dr/%d.png $dr/out.mpg`)
fig = plt.figure(figsize=(11,11))
	plt.subplot(2,2,3)
	plt.imshow(phix, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
	plt.xlabel("true lensing potential")
	plt.subplot(2,2,1)
	plt.imshow(phix_sum/cntr, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
	plt.xlabel("estimated lensing potential")
	plt.subplot(2,2,2)
	plt.imshow(tildetx_lr_sum/cntr, interpolation = "none", vmin=minimum(ytx),vmax=maximum(ytx)) 
	plt.xlabel("estimated unlensing CMB")
	plt.subplot(2,2,4)
	plt.imshow(ytx, interpolation = "none", vmin=minimum(ytx),vmax=maximum(ytx)) 
	plt.xlabel("data")
	plt.show()
run(`open $dr/out.mpg`)
run(`rm -fr $dr`)
end



