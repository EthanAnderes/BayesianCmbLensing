# julia scripts/example3.jl 
const dualMessanger = false 
const maskupC  = 3000.0  #l_max for cmb
const maskupP  = 500.0  #l_max for for phi
const hrfactor = 2.0
const pixel_size_arcmin = 5.0
const n = 2.0^9
const beamFWHM = 0.0
const nugget_at_each_pixel = (5.0)^2
const seed = rand(1:1000000)
const scriptname = "example3"
const savepath = joinpath("simulations", "$(scriptname)_$seed") # savepath = joinpath("simulations", "test") 
simnotes = """
low noise
larger pixels
"""

#-----------------------------------
# load modules and functions
#------------------------------------
push!(LOAD_PATH, pwd()*"/src")
using Interp
require("cmb.jl")
require("fft.jl")
require("funcs.jl") # use reload after editing funcs.jl

#---------------------------------------
# Set the parameters of the run 
#-----------------------------------------
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

#-----------------------------------------
# Simulate: ytx, maskvarx, phix, tildetx
#-----------------------------------------
srand(seed)
ytk_nomask, tildetk, phix, tx_hr = simulate_start(parlr);
phik = fft2(phix, parlr)
maskboolx =  falses(size(phix))
maskvarx = parlr.nugget_at_each_pixel .* ones(size(phix))
maskvarx[maskboolx] = Inf
ytx = ifft2r(ytk_nomask, parlr)
ytx[maskboolx] = 0.0
ytk = fft2(ytx, parlr)

#-----------------------------------------
#  set the save directory and save stuff 
#-----------------------------------------
isdir(savepath) && run(`rm -r $savepath`) 
run(`mkdir $savepath`)
run(`cp src/funcs.jl $savepath/funcs_save.jl`)
run(`cp scripts/$scriptname.jl $savepath/$(scriptname)_save.jl`)
run(`echo $simnotes` |> "$savepath/simnotes.txt")
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


#-----------------------------------------
#  initalized and run the gibbs 
#-----------------------------------------
tx_hr_curr    = zeros(parhr.grd.x)
tbarx_hr_curr = zeros(parhr.grd.x)
ttx_hr_curr   = zeros(parhr.grd.x)
phik_curr    = zeros(phik)
tildetx_lr_curr = zeros(parlr.grd.x) 
tildetx_hr_curr = zeros(parhr.grd.x) 
tx_hr_curr_sum = zeros(tx_hr_curr)
phik_curr_sum = zeros(phik)
acceptclk = [1 for k=1:10] #initialize acceptance record
bglp = 0

# srand(seedloop) # use this only if you are running multiple session with the same input 
while true
	# use phik_curr from previous iteration to simluate the unlensed CMB: tx_hr_curr
	if dualMessanger 
		if bglp % 100 == 0 phidx1_hr, phidx2_hr, phidx1_lr, phidx2_lr = gibbspass_coold!(tx_hr_curr, tbarx_hr_curr, phik_curr, ytx, maskvarx, parlr, parhr, 600) end
    	phidx1_hr, phidx2_hr, phidx1_lr, phidx2_lr = gibbspass_d!(tx_hr_curr, tbarx_hr_curr, phik_curr, ytx, maskvarx, parlr, parhr, 400)
	else  
		if  bglp % 100 == 0   phidx1_hr, phidx2_hr, phidx1_lr, phidx2_lr = gibbspass_coolt!(tx_hr_curr, ttx_hr_curr, phik_curr, ytx, maskvarx, parlr, parhr, 600) end
 		phidx1_hr, phidx2_hr, phidx1_lr, phidx2_lr = gibbspass_t!(tx_hr_curr, ttx_hr_curr, phik_curr, ytx, maskvarx, parlr, parhr, 400) 
	end
	tildetx_hr_curr[:] = spline_interp2(
		parhr.grd.x, 
		parhr.grd.y, 
		tx_hr_curr, 
		parhr.grd.x + phidx1_hr, 
		parhr.grd.y + phidx2_hr
		)
	
	# shock the system with a gradient update if the accpetence rate falls below 20%
	if  (bglp <= 4) | false  # (bglp <= 4) | (countnz(acceptclk[end-9:end]) <= 1) 
		gradupdate!(phik_curr, tildetx_hr_curr, parlr, parhr) 
		push!(acceptclk, 2)
	else 	
		push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr))
		push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr))
		push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr))
	end

	# update blgp, etc...
	if bglp % 5 == 1
		writecsv("$savepath/tildetx_lr_curr_$bglp.csv", tildetx_hr_curr[1:int(hrfactor):end,1:int(hrfactor):end])
		writecsv("$savepath/phix_curr_$bglp.csv", ifft2r(phik_curr, parlr))
		writecsv("$savepath/acceptclk.csv", acceptclk)
	end
	phik_curr_sum += phik_curr
	tx_hr_curr_sum += tx_hr_curr 
    bglp += 1
end

