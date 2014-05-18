# julia scripts/scriptNew.jl
simnotes = """
low noise
small pixels
"""
const scriptname = "scriptNew"
const percentNyqForC = 0.5 # used for T l_max
const numofparsForP  = 1000  # used for P l_max
const hrfactor = 2.0
const pixel_size_arcmin = 1.0
const n = 2.0^10
const beamFWHM = 0.0
const nugget_at_each_pixel = (6.0)^2
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
const seed = rand(1:1000000)
const savepath = joinpath("simulations", "$(scriptname)_$seed") 



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
srand(seed)
ytk_nomask, tildetk, phix, tx_hr = simulate_start(parlr);
phik = fft2(phix, parlr)
maskboolx =  falses(size(phix))
maskvarx = parlr.nugget_at_each_pixel .* ones(size(phix))
maskvarx[maskboolx] = Inf
ytx = ifft2r(ytk_nomask, parlr)
ytx[maskboolx] = 0.0
ytk = fft2(ytx, parlr)




# -----------  set the save directory and save stuff 
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





# ------------------ initalized and run the gibbs 
tx_hr_curr      = zeros(parhr.grd.x)
tbarx_hr_curr   = zeros(parhr.grd.x)
ttx_hr_curr     = zeros(parhr.grd.x)
phik_curr       = zeros(phik)
tildetx_lr_curr = zeros(parlr.grd.x) 
tildetx_hr_curr = zeros(parhr.grd.x) 
tx_hr_curr_sum  = zeros(tx_hr_curr)
phik_curr_sum   = zeros(phik)
acceptclk       = [1 for k=1:10] #initialize acceptance record


bglp = 0
while true
	#  ------ use phik_curr from previous iteration to simluate the unlensed CMB: tx_hr_curr
	#if  bglp % 100 == 0  
		phidx1_hr, phidx2_hr, phidx1_lr, phidx2_lr = gibbspass_coolt!(
			tx_hr_curr, 
			ttx_hr_curr, 
			phik_curr, 
			ytx, 
			maskvarx, 
			parlr, 
			parhr, 
			400, 
			maskupC
		) 
	#end
 	# phidx1_hr, phidx2_hr, phidx1_lr, phidx2_lr = gibbspass_t!(
 	# 	tx_hr_curr, 
 	# 	ttx_hr_curr, 
 	# 	phik_curr, 
 	# 	ytx, 
 	# 	maskvarx, 
 	# 	parlr, 
 	# 	parhr, 
 	# 	400, 
 	# 	maskupC
 	# 	) 
	tildetx_hr_curr[:] = spline_interp2(
		parhr.grd.x, 
		parhr.grd.y, 
		tx_hr_curr, 
		parhr.grd.x + phidx1_hr, 
		parhr.grd.y + phidx2_hr
		)
	
	#  ------ gradient updates at the start
	if  bglp <= 4
		gradupdate!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_grad) 
		push!(acceptclk, 2)
	else 	
		push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc))
		push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc))
		push!(acceptclk, hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc))
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

