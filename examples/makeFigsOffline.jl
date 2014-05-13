#=
include("examples/makeFigsOffline.jl")
=#
specc 	   = true # plot the spectral coverage
pcorr 	   = false # plot the empirical cross correlation
onedslice  = true # plot the 1-d slices of phi
acc 	   = false # take a look at the accpetence rate
imagsli    = false # look at the images one by one
mvie 	   = false # make movies and average estimate

krang = 501:(5):5000 # range of samples we are looking at

using PyCall 
@pyimport matplotlib.pyplot as plt
push!(LOAD_PATH, "../../src")
using Interp
require("cmb.jl")
require("fft.jl")


#---------------------------------------------------
# plot the spectral coverage
#-----------------------------------------
if specc
	maskupC  = 3000.0  #l_max for cmb
	maskupP  = 1000.0  #l_max for for phi
	pixel_size_arcmin = 2.0
	n = 2.0^9
	beamFWHM = 0.0
	nugget_at_each_pixel = (4.0)^2
	d = 2
	bin_mids = 16:15:700
	
	parlr = setpar(
		pixel_size_arcmin, 
		n, 
		beamFWHM, 
		nugget_at_each_pixel, 
		maskupC, 
		maskupP,
		"../../src"
	);
	dirac_0 = 1/parlr.grd.deltk^d 
	el = parlr.ell[10:1000] 
	cpl = parlr.CPell2d[10:1000]
	ctl = parlr.CTell2d[10:1000]
	mte =  el.^4
	mtr = parlr.grd.r.^2
	
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
		if isfile("phix_curr_$k.csv") & isfile("tildetx_lr_curr_$k.csv")
			phik_curr    = fft2(readcsv("phix_curr_$k.csv"), parlr) 
			tildetk_curr = fft2(readcsv("tildetx_lr_curr_$k.csv"), parlr)
			push!(phik_pwr_smpl,   binpower(mtr .* phik_curr, parlr.grd.r, bin_mids))
			push!(tildek_pwr_smpl, binpower(sqrt(mtr) .* tildetk_curr, parlr.grd.r, bin_mids))
			cntr += 1
		end
	end
	# squish to matrix
	phik_pwr_smpl = hcat(phik_pwr_smpl...)
	tildek_pwr_smpl = hcat(tildek_pwr_smpl...)
	# then split by the rows
	phb = Array{Float64,1}[phik_pwr_smpl[k,:][:]   for k=1:length(bin_mids)]
	tib = Array{Float64,1}[tildek_pwr_smpl[k,:][:] for k=1:length(bin_mids)]
	# now phb[2] shoudl be the samples over bin bin_mids[2]
	
	# here is the truth
	phik =   fft2(readcsv("phix.csv"), parlr)
	tildek = fft2(readcsv("tildetx_lr.csv"), parlr)
	
	phb_truth = binpower(mtr .* phik, parlr.grd.r, bin_mids)
	tib_truth = binpower(sqrt(mtr) .* tildek, parlr.grd.r, bin_mids)
	
	plt.figure(figsize=(15,5))
	plt.plot(el, dirac_0 .* mte .* cpl, "-k")
	plt.plot(bin_mids, phb_truth, "or", label = "truth")
	rtcuts  = collect(bin_mids +  step(bin_mids) / 2)  
	lftcuts = collect(bin_mids -  step(bin_mids) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	plt.errorbar(
		bin_mids,
		map(median,  phb),
		xerr = Array{Float64,1}[bin_mids-lftcuts, rtcuts-bin_mids],  
		yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), phb), map(x-> quantile(x,0.975)-median(x), phb)],
		fmt="*b",
		label = "posterior"
	)
	plt.legend()
	plt.show()


	plt.figure(figsize=(15,5))
	plt.plot(el, dirac_0 .* sqrt(mte) .* ctl, "-k")
	plt.plot(bin_mids, tib_truth, "or", label = "truth")
	rtcuts  = collect(bin_mids +  step(bin_mids) / 2)  
	lftcuts = collect(bin_mids -  step(bin_mids) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	plt.errorbar(
		bin_mids,
		map(median,  tib),
		xerr = Array{Float64,1}[bin_mids-lftcuts, rtcuts-bin_mids],  
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
	maskupC  = 3000.0  #l_max for cmb
	maskupP  = 1000.0  #l_max for for phi
	pixel_size_arcmin = 2.0
	n = 2.0^9
	beamFWHM = 0.0
	nugget_at_each_pixel = (4.0)^2
	d = 2
	bin_mids = 20:20:900
	
	parlr = setpar(
		pixel_size_arcmin, 
		n, 
		beamFWHM, 
		nugget_at_each_pixel, 
		maskupC, 
		maskupP,
		"../../src"
	);
	dirac_0 = 1/parlr.grd.deltk^d 
	el = parlr.ell[10:1000] 
	cpl = parlr.CPell2d[10:1000]
	mte =  el.^4
	mtr = parlr.grd.r.^2
	
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
	
	phik =   fft2(readcsv("phix.csv"), parlr)
	tildek = fft2(readcsv("tildetx_lr.csv"), parlr)
	
	phik_cov_smpl = Array{Float64,1}[]
	tildek_cov_smpl = Array{Float64,1}[]
	cntr = 0
	for k in krang
		if isfile("phix_curr_$k.csv") & isfile("tildetx_lr_curr_$k.csv")
			phik_curr    = fft2(readcsv("phix_curr_$k.csv"), parlr) 
			tildetk_curr = fft2(readcsv("tildetx_lr_curr_$k.csv"), parlr)
			pnxt   = binave(mtr .* phik_curr .* conj(phik), parlr.grd.r, bin_mids)
			pnxt ./= binpower(sqrt(mtr) .* phik_curr, parlr.grd.r, bin_mids) |> sqrt
			pnxt ./= binpower(sqrt(mtr) .* phik, parlr.grd.r, bin_mids) |> sqrt
			tnxt   = binave(mtr .* tildetk_curr .* conj(tildek), parlr.grd.r, bin_mids)
			tnxt ./= binpower(sqrt(mtr) .* tildetk_curr, parlr.grd.r, bin_mids) |> sqrt
			tnxt ./= binpower(sqrt(mtr) .* tildek, parlr.grd.r, bin_mids) |> sqrt
			push!(phik_cov_smpl, real(pnxt))
			push!(tildek_cov_smpl, real(tnxt))
			cntr += 1
		end
	end
	# squish to matrix
	phik_cov_smpl = hcat(phik_cov_smpl...)
	tildek_cov_smpl = hcat(tildek_cov_smpl...)
	# then split by the rows
	phb = Array{Float64,1}[phik_cov_smpl[k,:][:]   for k=1:length(bin_mids)]
	tib = Array{Float64,1}[tildek_cov_smpl[k,:][:] for k=1:length(bin_mids)]
	# now phb[2] shoudl be the samples over bin bin_mids[2]
	
	# for phi
	plt.figure(figsize=(15,5))
	rtcuts  = collect(bin_mids +  step(bin_mids) / 2)  
	lftcuts = collect(bin_mids -  step(bin_mids) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	plt.errorbar(
		bin_mids,
		map(median,  phb),
		xerr = Array{Float64,1}[bin_mids-lftcuts, rtcuts-bin_mids],  
		yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), phb), 
								map(x-> quantile(x,0.975)-median(x), phb)],
		fmt="*b",
		label = "empirical correlation with truth"
	)
	plt.plot(collect(bin_mids), zero(bin_mids), ":k")
	plt.legend()
	plt.show()
	# for T
	plt.figure(figsize=(15,5))
	rtcuts  = collect(bin_mids +  step(bin_mids) / 2)  
	lftcuts = collect(bin_mids -  step(bin_mids) / 2)  
	lftcuts[1] = 0.1 # extend the left boundary all the way down, but not zero
	plt.errorbar(
		bin_mids,
		map(median,  tib),
		xerr = Array{Float64,1}[bin_mids-lftcuts, rtcuts-bin_mids],  
		yerr = Array{Float64,1}[map(x-> median(x)-quantile(x,0.025), tib), 
								map(x-> quantile(x,0.975)-median(x), tib)],
		fmt="*b",
		label = "empirical correlation with truth"
	)
	plt.plot(collect(bin_mids), zero(bin_mids), ":k")
	plt.legend()
	plt.show()
end 


#---------------------------------------------------
# plot the 1-d slices of phi
#-----------------------------------------
if onedslice
	propslice = 0.2 # btwn 0 and 1
	phix_slice_samples = Array{Float64,1}[]
	tildex_slice_samples = Array{Float64,1}[]
	cntr = 0
	for k in krang
		if isfile("phix_curr_$k.csv") & isfile("tildetx_lr_curr_$k.csv")
			phix_curr = readcsv("phix_curr_$k.csv")
			tildetx_curr = readcsv("tildetx_lr_curr_$k.csv")
			push!(phix_slice_samples, phix_curr[int(end*propslice),:][:])
			push!(tildex_slice_samples, tildetx_curr[int(end*propslice),:][:])
			cntr += 1
		end
	end
	phix_slice_samples = hcat(phix_slice_samples...)
	tildex_slice_samples = hcat(tildex_slice_samples...)
	
	# files I can upload directly
	phix = readcsv("phix.csv")[int(end*propslice),:][:]
	tildex = readcsv("tildetx_lr.csv")[int(end*propslice),:][:]
	qex = readcsv("qex_lr.csv")[int(end*propslice),:][:]
	x = readcsv("x.csv")[int(end*propslice),:][:]
	varx = readcsv("maskvarx.csv")[int(end*propslice),:][:]
	maskmin, maskmax = minimum(x[varx.==Inf]), maximum(x[varx.==Inf])
	
	# plot phix
	plt.plot(x, phix_slice_samples[:,1], color="blue", alpha=0.4, label="posterior samples")
	plt.plot(x, phix_slice_samples[:,2:end], color="blue", alpha=0.4)
	plt.plot(x, phix, color = "red",  linewidth=2.5, linestyle="-", label="truth")
	plt.plot(x, qex, color = "green",  linewidth=2.5, linestyle="--", label="quadratic estimate")
	plt.plot(x,  mean(phix_slice_samples,2), color = "black", linewidth=2.5, linestyle="--",  alpha=0.8, label="posterior mean")
	plt.plot(x, zero(phix), color = "black",  linestyle=":")
	plt.legend(loc="upper right")
	#plt.axis("tight")
	plt.annotate("mask",
	         xy=(maskmin, 0), xycoords="data",
	         xytext=(-50, -50), textcoords="offset points", fontsize=16,
	         arrowprops={:arrowstyle=>"->"})
	plt.title("Lensing potential posterior samples")
	plt.xlabel("radians")
	plt.axvspan(maskmin, maskmax,  facecolor="0.5", alpha=0.3, label="mask region")
	plt.show()
	
	
	#plot tildex
	plt.plot(x, tildex_slice_samples[:,1], color="blue", alpha=0.4, label="posterior samples")
	plt.plot(x, tildex_slice_samples[:,2:end], color="blue", alpha=0.4)
	plt.plot(x, tildex, color = "red",  linewidth=2.5, linestyle="-", label="truth")
	plt.plot(x,  mean(tildex_slice_samples,2), color = "black", linewidth=2.5, linestyle="--",  alpha=0.8, label="posterior mean")
	plt.plot(x, zero(tildex), color = "black",  linestyle=":")
	plt.legend(loc="upper right")
	#plt.axis("tight")
	plt.annotate("mask",
	         xy=(maskmin, 0), xycoords="data",
	         xytext=(-50, -50), textcoords="offset points", fontsize=16,
	         arrowprops={:arrowstyle=>"->"})
	plt.title("Lensed CMB posterior samples")
	plt.xlabel("radians")
	plt.axvspan(maskmin, maskmax,  facecolor="0.5", alpha=0.3, label="mask region")
	plt.show()
end 


#---------------------------------------------------
# take a look at the accpetence rate
#-----------------------------------------
if acc 
	accptrec = readcsv("acceptclk.csv")
	
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
	phix = readcsv("phix.csv")
	tildex = readcsv("tildetx_lr.csv")
	ytx  = readcsv("ytx.csv")
	for k in krang
		if isfile("phix_curr_$k.csv") & isfile("tildetx_lr_curr_$k.csv")
			phix_curr = readcsv("phix_curr_$k.csv")
			tildetx_curr = readcsv("tildetx_lr_curr_$k.csv")
			plt.figure(figsize=(12,12))
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


#---------------------------------
#  make movies and average estimate
# -----------------------------------------
if mvie 
tm = "trashthismovie"  #string(time_ns()) 
dr = "/tmp/" * tm * "mov"
isdir(dr) || mkdir(dr)

function saveMovieFram(dr, num, phix, phix_curr, ytx, tildetx_lr_curr)
	# save to dr
	fig = plt.figure(figsize=(11,11))
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
phix = readcsv("phix.csv")
ytx  = readcsv("ytx.csv")
phix_sum = zero(phix)
tildetx_lr_sum = zero(phix)
cntr = 0
for k in krang
	if isfile("phix_curr_$k.csv") & isfile("tildetx_lr_curr_$k.csv")
		phix_curr = readcsv("phix_curr_$k.csv")
		tildetx_lr_curr = readcsv("tildetx_lr_curr_$k.csv")
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
# run(`rm -fr $dr`)
end



