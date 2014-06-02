#= 
	julia scripts/scriptNew.jl
=#
const scriptname = "scriptNew"
# const seed = Base.Random.RANDOM_SEED
const seed = 10000; srand(seed)
const savepath = joinpath("simulations", "$(scriptname)_MaskT_$(seed[1])") #<--change the directory name here
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
const scale_grad =  5.0e-3
const scale_hmc  =  1.0e-3



# ------------ load modules and functions
push!(LOAD_PATH, pwd()*"/src")
using Interp
require("cmb.jl")
require("fft.jl")
require("funcs.jl") # use reload after editing funcs.jl



# --------- generate cmb spectrum class for high res and low res
parlr = setpar(
	pixel_size_arcmin, 
	n, 
	beamFWHM, 
	nugget_at_each_pixel, 
	maskupC, 
	maskupP
);
parhr = setpar(
	pixel_size_arcmin./hrfactor, 
	hrfactor*n, 
	beamFWHM, 
	nugget_at_each_pixel, 
	maskupC, 
	maskupP
);



# -------- Simulate data: ytx, maskvarx, phix, tildetx
ytk_nomask, tildetk, phix, tx_hr = simulate_start(parlr);
phik = fft2(phix, parlr)
# maskboolx =  falses(size(phix))
tmpdo = maximum(parlr.grd.x)*0.3
tmpup = maximum(parlr.grd.x)*0.4
maskboolx = tmpdo .<= parlr.grd.x .<= tmpup
maskvarx  = parlr.nugget_at_each_pixel .* ones(size(parlr.grd.x))
maskvarx[maskboolx] = Inf
ytx = ifft2r(ytk_nomask, parlr)
ytx[maskboolx] = 0.0
ytk = fft2(ytx, parlr)

# the following makes maskvarx_cool used in a cooling step
decay(x) = 1 ./ (abs(x).^(1/4))
maskvarx_cool = maskvarx + decay(parlr.grd.x .- tmpup) + decay(parlr.grd.x .- tmpdo)
maskvarx_cool += decay(parlr.grd.x .- tmpup)
maskvarx_cool += decay(parlr.grd.x .- tmpdo)
# plot(parlr.nugget_at_each_pixel * ones(512))
# plot(parlr.nugget_at_each_pixel * zeros(512))
# plot(maskvarx_cool[1,:].')
# plot(0.3*512, parlr.nugget_at_each_pixel, "*")
# plot(0.4*512, parlr.nugget_at_each_pixel, "*")
# axis([0, 500, 0, 40])



# -----------  set the save directory and save stuff 
isdir(savepath) && run(`rm -r $savepath`) 
run(`mkdir $savepath`)
run(`cp src/funcs.jl $savepath/funcs_save.jl`)
run(`cp scripts/$scriptname.jl $savepath/$(scriptname)_save.jl`)
writecsv("$savepath/seed.csv", seed)
writecsv("$savepath/x.csv", parlr.grd.x)
writecsv("$savepath/y.csv", parlr.grd.y)
writecsv("$savepath/k1.csv", parlr.grd.k1)
writecsv("$savepath/k2.csv", parlr.grd.k2)
writecsv("$savepath/ytx.csv", ytx)
writecsv("$savepath/phix.csv", phix)
writecsv("$savepath/tildetx_lr.csv", ifft2r(tildetk, parlr))
writecsv("$savepath/tx_lr.csv", tx_hr[1:hrfactor:end, 1:hrfactor:end] )
writecsv("$savepath/maskvarx.csv", maskvarx)
# print out the quadratic estimate for comparision
al_lr = ttk_al(parlr)
qex_lr = ifft2r(ttk_est(ytk_nomask, parlr) .* (parlr.cPP ./ (2.0 .* al_lr + parlr.cPP)) , parlr)
writecsv("$savepath/qex_lr.csv", qex_lr)	





# ------------------ initalized and run the gibbs 
function gibbsloop(its, parhr, parlr, ytx, maskvarx)
	tx_hr_curr      = zero(parhr.grd.x)
	ttx_hr_curr     = zero(parhr.grd.x)
	phidx1_hr, phidx2_hr, phidx1_lr, phidx2_lr = zero(parhr.grd.x), zero(parhr.grd.x), zero(parlr.grd.x), zeros(parlr.grd.x)
	phik_curr       = zero(fft2(ytx, parlr))
	tildetx_hr_curr = zero(parhr.grd.x) 
	acceptclk       = [1] #initialize acceptance record

	for bglp = 1:its
		#  ------ use phik_curr from previous iteration to simluate the unlensed CMB: tx_hr_curr
		phidx1_hr[:], phidx2_hr[:], phidx1_lr[:], phidx2_lr[:] = gibbspass_t!(
			tx_hr_curr, ttx_hr_curr, 
			phik_curr, ytx, maskvarx, 
			parlr, parhr, 
			linspace(parhr.grd.deltk, 2maskupC, 500) 
		)
		
		tildetx_hr_curr[:] = spline_interp2(
			parhr.grd.x, parhr.grd.y, tx_hr_curr, 
			parhr.grd.x + phidx1_hr, parhr.grd.y + phidx2_hr
		)
		
		if bglp <= 5 
			gradupdate!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_grad) 
		else
			push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc))
			push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc))
			push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc))
		end
	
		writecsv("$savepath/tildetx_lr_curr_$bglp.csv", tildetx_hr_curr[1:int(hrfactor):end,1:int(hrfactor):end])
		writecsv("$savepath/phix_curr_$bglp.csv", ifft2r(phik_curr, parlr))
		writecsv("$savepath/acceptclk.csv", acceptclk)	
	end # for
end # function
gibbsloop(2000, parhr, parlr, ytx, maskvarx)