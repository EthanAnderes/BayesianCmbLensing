

#--- set the seed or generate a random one
# const seed = Base.Random.RANDOM_SEED
const seed = Uint32[2657077506, 531413685, 532641760, 1678774672] # big bump
srand(seed)



# ---- set directory name for the simulation
const scriptname = "scriptTrack"
const savepath = joinpath(pwd(), "simulations", "$(scriptname)_$(seed[1])") #<--change the directory name here



# ---- parameters of the simulation run
const maxiter = 5_000  # sets the maximum number of gibbs iterations for each worker
const percentNyqForC = 0.5   # used for T l_max
const numofparsForP  = 1500  # used for P l_max
const hrfactor = 2.0
const pixel_size_arcmin = 2.0
const n = 2.0^9
const beamFWHM = 0.0
const nugget_at_each_pixel = (4.0)^2
begin  
	local deltx =  pixel_size_arcmin * π / (180 * 60) # rads
	local period = deltx * n # side length in rads
	local deltk =  2 * π / period
	local nyq = (2 * π) / (2 * deltx)
	const maskupP  = √(deltk^2 * numofparsForP / π)  #l_max for for phi
	const maskupC  = min(9000.0, percentNyqForC * (2 * π) / (2 * pixel_size_arcmin * π / (180*60))) #l_max for for phi
	println("muK_per_arcmin = $(sqrt(nugget_at_each_pixel * (pixel_size_arcmin^2)))") # muK per arcmin
	println("maskupP = $maskupP") # muK per arcmin
	println("maskupC = $maskupC") # muK per arcmin
	println("$(seed[1])") # muK per arcmin
end
const scale_grad =  2.0e-3
const scale_hmc  =  2.0e-3



# ------------ load modules and functions
push!(LOAD_PATH, pwd()*"/src")
using Interp
FFTW.set_num_threads(15)
BLAS.blas_set_num_threads(15)
require("cmb.jl")
require("fft.jl")
require("funcs.jl") # use reload after editing funcs.jl
require("Interp.jl")



# --------- generate cmb spectrum class for high res and low res
parlr = setpar(
	pixel_size_arcmin, n, beamFWHM, nugget_at_each_pixel, 
	maskupC, maskupP
)
parhr = setpar(
	pixel_size_arcmin./hrfactor, hrfactor*n, beamFWHM, nugget_at_each_pixel, 
	maskupC, maskupP
)



# -------- Simulate data: ytx, maskvarx, phix, tildetx
ytk_nomask, tildetk, phix, tx_hr = simulate_start(parlr);
phik = fft2(phix, parlr)
tmpdo = maximum(parlr.grd.x)*0.3
tmpup = maximum(parlr.grd.x)*0.4
maskboolx = tmpdo .<= parlr.grd.x .<= tmpup
maskvarx  = parlr.nugget_at_each_pixel .* ones(size(parlr.grd.x))
maskvarx[maskboolx] = Inf
ytx = ifft2r(ytk_nomask, parlr)
ytx[maskboolx] = 0.0
ytk = fft2(ytx, parlr)


# -----------  set the save directory and save stuff 
isdir(savepath) && run(`rm -rf $savepath`) 
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
writecsv("$savepath/tx_lr.csv", tx_hr[1:3:end, 1:3:end] )
writecsv("$savepath/maskvarx.csv", maskvarx)

# ------------------ initalized and run the gibbs 
function gibbsloop(its, parhr, parlr, ytx, maskvarx, maskupC, savepath, scale_grad, scale_hmc, hrfactor, record_run)
	acceptclk   = [1] #initialize acceptance record
	tx_hr_curr  = zero(parhr.grd.x)
	ttx         = zero(parhr.grd.x)
	p1hr, p2hr  = zero(parhr.grd.x), zero(parhr.grd.x)
	phik_curr   = zero(fft2(ytx, parlr))
	phix_curr   = zero(ytx)
	tx_lr_curr  = zero(ytx)
	tildetx_hr_curr = zero(parhr.grd.x) 

	for bglp = 1:its 
		tic()
		# ----- update tildetx_hr_curr
		if bglp % 100 == 1
			p1hr[:], p2hr[:] = gibbspass_t!(tx_hr_curr, ttx, phik_curr, ytx, 
				maskvarx, parlr, parhr, repstretch(2.0, 1.2maskupC, 75, 1000) 
			)
		end
		p1hr[:], p2hr[:] = gibbspass_t!(tx_hr_curr, ttx, phik_curr, ytx, 
			maskvarx, parlr, parhr, fill(Inf, 400)
		)

		tildetx_hr_curr[:] = spline_interp2(
			parhr.grd.x, parhr.grd.y, tx_hr_curr, 
			parhr.grd.x + p1hr, parhr.grd.y + p2hr
		)
		
		# ----- update phik_curr
		push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc))
		
		# ----- now save
		phix_curr[:]  = ifft2r(phik_curr, parlr)
		tx_lr_curr[:] =  tx_hr_curr[1:int(hrfactor):end,1:int(hrfactor):end]
		record_run = hcat(record_run,
			[tx_lr_curr[400,300], 
			 tx_lr_curr[50,50], 
			 phix_curr[400,300], 
			 phix_curr[50,50], 
			 phik_curr[10,10], 
			 phik_curr[4,7]])
		if bglp % 50 == 1
			writecsv("$savepath/trace.csv", record_run)
			writecsv("$savepath/phix_curr_$bglp.csv", phix_curr)
			writecsv("$savepath/acceptclk.csv", acceptclk)	
		end # if
		toc()
	end # for
end # function


tx_lr = tx_hr[1:3:end,1:3:end]
record_run = [tx_lr[400,300], 
			 tx_lr[50,50], 
			 phix[400,300], 
			 phix[50,50], 
			 phik[10, 10], 
			 phik[4, 7]]
gibbsloop(maxiter, parhr, parlr, ytx, maskvarx, maskupC, savepath, scale_grad, scale_hmc, hrfactor, record_run)



