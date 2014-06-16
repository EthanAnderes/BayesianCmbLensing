const seed = 10000; srand(seed)
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
end
push!(LOAD_PATH, pwd()*"/src")
using Interp, PyPlot
require("cmb.jl")
require("fft.jl")
require("funcs.jl") # use reload after editing funcs.jl
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
# maskboolx =  falses(size(phix))
tmpdo = maximum(parlr.grd.x)*0.3
tmpup = maximum(parlr.grd.x)*0.4
maskboolx = tmpdo .<= parlr.grd.x .<= tmpup
maskvarx  = parlr.nugget_at_each_pixel .* ones(size(parlr.grd.x))
maskvarx[maskboolx] = Inf
ytx = ifft2r(ytk_nomask, parlr)
ytx[maskboolx] = 0.0
ytk = fft2(ytx, parlr)


tx_hr_curr  = zero(parhr.grd.x)
ttx  = zero(parhr.grd.x)
tttx  = zero(parhr.grd.x)
p1hr, p2hr  = zero(parhr.grd.x), zero(parhr.grd.x)
phik_curr   = zero(fft2(ytx, parlr))
tildetx_hr_curr = zero(parhr.grd.x) 

cool =  linspace(100, 3000, 50)
p1hr[:], p2hr[:] = gibbspass_td!(tx_hr_curr, ttx, tttx, phik_curr, ytx, 
      maskvarx, parlr, parhr, cool
)


#p1hr[:], p2hr[:] = gibbspass_d!(tx_hr_curr, tttx, phik_curr, ytx, 
#      maskvarx, parlr, parhr, linspace(100, 3000, 25)
#)
#p1hr[:], p2hr[:] = gibbspass_t!(tx_hr_curr, tttx, phik_curr, ytx, 
#      maskvarx, parlr, parhr, fill(Inf, 25)
#)


imshow(tx_hr_curr)










#--- test messenger algorithms
const seed = 10000; srand(seed)
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
# ------------ load modules and functions
push!(LOAD_PATH, pwd()*"/src")
using Interp, PyPlot
require("cmb.jl"); require("fft.jl"); require("funcs.jl") # use reload after editing funcs.jl
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
# maskboolx =  falses(size(phix))
tmpdo = maximum(parlr.grd.x)*0.3
tmpup = maximum(parlr.grd.x)*0.4
maskboolx = tmpdo .<= parlr.grd.x .<= tmpup
maskvarx  = parlr.nugget_at_each_pixel .* ones(size(parlr.grd.x))
maskvarx[maskboolx] = Inf
ytx = ifft2r(ytk_nomask, parlr)
ytx[maskboolx] = 0.0
ytk = fft2(ytx, parlr)
#----- define gibbs
function gibbspass_t!(sx, tx, phik_curr, ytx, maskvarx, parlr, parhr, coolingVec = [Inf for k=1:100])
    phidx1_hr_curr, phidx2_hr_curr, phidx1_lr_curr, phidx2_lr_curr = phi_easy_approx(phik_curr, parlr, parhr)
    dx, Nx = embedd(ytx, phidx1_lr_curr, phidx2_lr_curr, maskvarx, parlr, parhr)
    # ------ pre-allocate space
    tk = fft2(tx, parhr)
    sk = fft2(sx, parhr)
    tpx = Array(Float64, size(tx))
    chng  = Array(Bool, size(tx))
    Tx    = 0.99 * minimum(Nx)
    barNx = Nx .- Tx
    d2k = parhr.grd.deltk * parhr.grd.deltk
    d2x = parhr.grd.deltx * parhr.grd.deltx
    delt0 = 1 / d2k
    Tk  = Tx * d2x # Tk is the spectrum, Tx is the pixelwise variance
    # ------ gibbs with cooling:) 
    for uplim in coolingVec
        λ =  (uplim > 8000.0) ? 1.0 : max(1.0, parhr.CTell2d[round(uplim)]/Tk)
        chng = parhr.grd.r .<= uplim
        # ---- update s
        tpx[:]   = 1 ./ (1 / (λ * Tk * delt0) .+ 1 ./ (parhr.cTT * delt0)) 
        sk[chng] = tpx[chng] .* (tk[chng] / (λ * Tk * delt0))  # wiener filter
        sk[chng]+= white(parhr)[chng] .* √(tpx)[chng]      # random fluctuation
        sx[:]    = ifft2r(sk, parhr) 
        # ---- update t
        # whatever updates are done, they are only done on the low ell multipoles
        barNxsmth = smooth_to_inf(barNx, uplim, parhr)
        tpx[:]   = 1 ./ (1 ./ barNxsmth .+ 1 / (λ * Tx)) 
        # can the following pointwise averaging be done in a spatially smooth way?
        tx_tmp   = tpx .* (dx ./ barNxsmth + sx ./ (λ * Tx)) # wiener filter...weighted ave of dx and sx.
        tx_tmp  += randn(size(tx)) .* √(tpx)             # random fluctuation
        tk_tmp   = fft2(tx_tmp, parhr)
        tk[chng] = tk_tmp[chng]
        tx[:]    = ifft2r(tk, parhr)
    end
    phidx1_hr_curr, phidx2_hr_curr
end
function gibbspass_d!(sx, sbarx, phik_curr, ytx, maskvarx, parlr, parhr, coolingVec = [Inf for k=1:100])
    phidx1_hr_curr, phidx2_hr_curr, phidx1_lr_curr, phidx2_lr_curr = phi_easy_approx(phik_curr, parlr, parhr)
    dx, Nx = embedd(ytx, phidx1_lr_curr, phidx2_lr_curr, maskvarx, parlr, parhr)
    d2k = parhr.grd.deltk * parhr.grd.deltk
    d2x = parhr.grd.deltx * parhr.grd.deltx
    delt0 = 1 / d2k
    for uplim in coolingVec
        λk = delt0 .* parhr.CTell2d[min(8000, round(uplim))] 
        λx = λk / (delt0 * d2x) 
        Sbark = delt0 .* parhr.cTT .- λk 
        Sbark[Sbark .< 0.0]= 0.0 
        # ---- update sbarx
        tmp      = λk .* Sbark ./ (Sbark .+ λk) # this can work when Sbark is zero
        sbark    = tmp .* fft2(sx, parhr) ./ λk # wiener filter
        sbark   += white(parhr) .* √(tmp)    # fluctuation
        sbarx[:] = ifft2r(sbark, parhr)
        # --- update sx
        tmp    = 1 ./ (1 ./ Nx .+ 1 / λx)
        sx[:]  = tmp .* (dx ./ Nx  .+ sbarx / λx)  
        sx[:]  = vec(sx + randn(size(sx)) .* sqrt(tmp))
    end
    sk = fft2(sx, parhr)
    sk[parhr.grd.r .> coolingVec[end]] = 0.0
    sx[:] = ifft2r(sk, parhr)
    phidx1_hr_curr, phidx2_hr_curr
end
function smooth_to_inf(maskvarx, uplim, par) 
  if uplim >= 8000
    return maskvarx
  end 
  varx = copy(maskvarx)
  indinf = (varx .== Inf)
  varx[indinf] = minimum(varx) * 100 #/ log(λ) # exp(1/(λ-1))
  fwhm = (2 * pi) / uplim  # in rad scale
  sig  = fwhm / (2 * √(2 * log(2)))
  beam = exp(-(sig^2) * (par.grd.r.^2) / 2)
  vark = fft2(varx, par) .* beam
  varx[:] = ifft2r(vark, par)
  varx[indinf] = Inf
  varx
end
#plot(smooth_to_inf(maskvarx, 100, parlr)[1,:]')
#plot(smooth_to_inf(maskvarx, 500, parlr)[1,:]')
#plot(smooth_to_inf(maskvarx, 1000, parlr)[1,:]')
#plot(smooth_to_inf(maskvarx, 4000, parlr)[1,:]')
#plot(smooth_to_inf(maskvarx, 8000, parlr)[1,:]')



# ------------------ initalized and run the gibbs 
tx_hr_curr      = zero(parhr.grd.x)
ttx_hr_curr     = zero(parhr.grd.x)
p1hr, p2hr      = zero(parhr.grd.x), zero(parhr.grd.x)
phik_curr       = zero(fft2(ytx, parlr))
tildetx_hr_curr = zero(parhr.grd.x) 
acceptclk       = [1] # initialize acceptance record
# cool = [linspace(10, 100, 30), fill(Inf, 20)]
# cool = [fill(100, 30), fill(Inf, 20)]
cool = linspace(4parhr.grd.deltk, 2700, 50)
@time p1hr[:], p2hr[:] = gibbspass_t!(tx_hr_curr, ttx_hr_curr, phik_curr, ytx, 
  maskvarx, parlr, parhr, cool
);

figure(figsize=(11,4))
subplot(1,2,1)
imshow(tx_hr_curr)
colorbar()
subplot(1,2,2)
imshow(tx_hr)
colorbar()











#--------------------------
#    test alternating the dual and t messenger.
#--------------------------------
const seed = Base.Random.RANDOM_SEED
const percentNyqForC = 0.5 # used for T l_max
const numofparsForP  = 1500  # used for P l_max
const hrfactor = 2.0
const pixel_size_arcmin = 1.5
const n = 2.0^9
const beamFWHM = 0.0
const nugget_at_each_pixel = (5.0)^2
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
# ------------ load modules and functions
push!(LOAD_PATH, pwd()*"/src")
using Interp, PyPlot
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
maskboolx =  (maximum(parlr.grd.x)*0.3) .<= parlr.grd.x .<= (maximum(parlr.grd.x)*0.4) 
# maskboolx =  falses(size(phix))
maskvarx = parlr.nugget_at_each_pixel .* ones(size(phix))
maskvarx[maskboolx] = Inf
ytx = ifft2r(ytk_nomask, parlr)
ytx[maskboolx] = 0.0
ytk = fft2(ytx, parlr)


sx_t  = zero(ytx)
tx    = zero(ytx)
sx_d  = zero(ytx)
sbarx = zero(ytx)


function bindss!(sx_t, sx_d, parlr, ulim)
  sk_t = fft2(sx_t, parlr)
  sk_d = fft2(sx_d, parlr)
  iup = parlr.grd.r  .> ulim
  sk_t[iup] = sk_d[iup]
  sx_fuse = ifft2r(sk_t, parlr)
  sx_t[:] = sx_fuse[:]
  sx_d[:] = sx_fuse[:]
end

for k=1:10
  wsim_gibbs_t!(sx_t, tx,    ytx, maskvarx, parlr, Inf)
  wsim_gibbs_d!(sx_d, sbarx, ytx, maskvarx, parlr, maskupC/2)
  #bindss!(sx_t, sx_d, parlr, maskupC/8)
end



#---------------------------
#  check the signal to noise ratio for T
#------------------------------
const scriptname = "scriptNew"
const percentNyqForC = 0.5 # used for T l_max
const numofparsForP  = 1500  # used for P l_max
const hrfactor = 2.0
const pixel_size_arcmin = 1.5
const n = 2.0^9
const beamFWHM = 0.0
const nugget_at_each_pixel = (5.0)^2
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
using PyPlot
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
d=2
dirac_0 = 1/parlr.grd.deltk^d 
elp = parlr.ell[10:int(maskupP)] 
elt = parlr.ell[10:int(maskupC)] 
cpl = parlr.CPell2d[10:int(maskupP)]
ctl = parlr.CTell2d[10:int(maskupC)]
r2 = parlr.grd.r.^2
bin_mids_P = (parlr.grd.deltk*2):(parlr.grd.deltk):maskupP
bin_mids_T = (parlr.grd.deltk*2):(parlr.grd.deltk):maskupC

plot(elt, dirac_0 .* elt.^2 .* ctl, "-k")
plot(elt, dirac_0 .* elt.^2 .* nugget_at_each_pixel .* (parlr.grd.deltx^2))
# notice the signal to noise drops below 1.0 just before 3500




#------------------------------------
# test out the HMC sampler on Wiener filtering
#--------------------------------------------
const maskupC  = 3000.0  # l_max for cmb
const maskupP  = 1000.0  # l_max for for phi
const pixel_size_arcmin = 2.0
const n = 2.0^9
const beamFWHM = 0.0
const d = 2
const nugget_at_each_pixel = (4)^2
using PyCall 
@pyimport matplotlib.pyplot as plt
push!(LOAD_PATH, pwd()*"/src")
using Interp
require("cmb.jl")
require("fft.jl")
function hmc!(phik_curr, datax, parlr)
	d = 2
	dirac_0 = 1/parlr.grd.deltk^d 
    ePs = 1.0e-3 * rand()
    ulim =  30 
    posterior_spectrum  = 1.0 .* parlr.cNT .* parlr.cPP ./ (parlr.cNT .+ parlr.cPP) 
    mk = 1.0 ./ (posterior_spectrum) # make this inv var
    mk[parlr.pMaskBool] = 0.0
    phik_test = copy(phik_curr)
    rk   = white(parlr, d) .* sqrt(mk); # note that the variance of real(pk_init) and imag(pk_init) is dirac_0 * mk/2 # specrum of rk is mk
    grad, loglike   = grad_wlog(datax, phik_test, parlr)
    h_at_zero = 0.5 * sum( 0.5 .* abs2(rk[!parlr.pMaskBool]) ./ (dirac_0 .* mk[!parlr.pMaskBool] ./ 2)) - loglike 
    for HMCCounter = 1:ulim 
        loglike = lfrog!(phik_test, rk, datax, parlr, ePs, mk)
    end
    h_at_end = 0.5 * sum( 0.5 .* abs2(rk[!parlr.pMaskBool]) ./ (dirac_0 .* mk[!parlr.pMaskBool] ./ 2)) - loglike # the 0.5 is out front since only half the sum is unique
    prob_accept = minimum([1, exp(h_at_zero - h_at_end)])
    if rand() < prob_accept
        phik_curr[:] = phik_test
        println("Accept: prob_accept = $(round(prob_accept,4)), h_at_end = $(round(h_at_end)), h_at_zero = $(round(h_at_zero)), loglike = $(round(loglike))")
        return 1
    else
        println("Reject: prob_accept = $(round(prob_accept,4)), h_at_end = $(round(h_at_end)), h_at_zero = $(round(h_at_zero)), loglike = $(round(loglike))")
        return 0
    end
end
function lfrog!(phik_curr, rk, datax, parlr, ePs, mk)
    grad, loglike   = grad_wlog(datax, phik_curr, parlr)
    rk_halfstep =  rk +  ePs .* grad ./ 2.0
    inv_mk = 1./ (mk ./ 2.0)
    inv_mk[parlr.pMaskBool] = 0.0
    phik_curr[:] = phik_curr + ePs .* inv_mk .* rk_halfstep
    grad, loglike   = grad_wlog(datax, phik_curr, parlr)
    rk[:] = rk_halfstep + ePs .* grad ./ 2.0
    loglike
end
function grad_wlog(datax, phik_curr, parlr)
	d = 2
	dirac_0 = 1/parlr.grd.deltk^d 
 	datak = fft2(datax, parlr)
    #--------------- log likelihood --------
    loglike = 0.0
    maskforpsi = one(phik_curr)
    maskforpsi[parlr.pMaskBool] = 0.0
    for k=2:length(phik_curr)
        loglike += - 0.25 * abs2(maskforpsi[k] * phik_curr[k]) / (dirac_0 *  parlr.cPP[k] / 2.0)
        loglike += - 0.25 * abs2(datak[k] - phik_curr[k]) / (dirac_0 *  parlr.cNT[k] / 2.0)
    end
    #--------- gradient --------------------
    gradk  = - phik_curr  ./ (dirac_0 .*  parlr.cPP ./ 2.0)
    gradk +=   (datak - phik_curr) ./ (dirac_0 .*  parlr.cNT ./ 2.0)
    gradk[parlr.pMaskBool] = 0.0
    gradk, loglike
end
function white(par::SpectrumGrids, d) 
    (par.grd.deltx^(-d/2.0)) .* fft2(randn(size(par.grd.x)), par) # this has spectrum dirac_0, i.e. the real and imag have var 1/(2deltk^d)
end
function  simulate_start(par::SpectrumGrids)
	d = 2
    phik   = white(par, d) .* sqrt(par.cPP) 
    noisek = white(par, d) .* sqrt(par.cNT) 
    datak = phik + noisek
    datax = ifft2r(datak, par)
    datax, phik
end
function gradupdate!(phik_curr, datax, parlr)
    for cntr = 1:30
        gradk, loglike = grad_wlog(datax, phik_curr, parlr)
        println(loglike)
        phik_curr[:]  =  phik_curr + gradk .* 1.0e-4 .* parlr.cNT .* parlr.cPP ./ (parlr.cNT .+ parlr.cPP) 
    end
end
function smooth_heavy(x, height1, height2, location, transition_sharpness)
    zeroto1 = 0.5 .+ 0.5 .* tanh(transition_sharpness .* (x.-location))
    (height2 .- height1) .* zeroto1 .+ height1
end
macro imag(ex)
  #the quote allows us to splice in ex with $
  quote  
      plt.imshow($ex,interpolation = "nearest")
      plt.colorbar()
      plt.show()
  end
end
macro imag2(ex1, ex2)
  #the quote allows us to splice in ex with $
  quote  
      plt.figure(figsize=(12,6))
      plt.subplot(1,2,1)
      plt.imshow($ex1,interpolation = "nearest")
      plt.colorbar()
      plt.subplot(1,2,2)
      plt.imshow($ex2,interpolation = "nearest")
      plt.colorbar()
      plt.show()
  end
end
macro imag4(ex1, ex2, ex3, ex4)
  #the quote allows us to splice in ex with $
  quote  
      plt.figure(figsize=(12,12))
      plt.subplot(2,2,1)
      plt.imshow($ex1,interpolation = "nearest")
      plt.colorbar()
      plt.subplot(2,2,2)
      plt.imshow($ex2,interpolation = "nearest")
      plt.colorbar()
      plt.subplot(2,2,3)
      plt.imshow($ex3,interpolation = "nearest")
      plt.colorbar()
      plt.subplot(2,2,4)
      plt.imshow($ex4,interpolation = "nearest")
      plt.colorbar()
      plt.show()
  end
end

parlr = setpar(pixel_size_arcmin, n, beamFWHM, nugget_at_each_pixel, maskupC, maskupP);
datax, phik = simulate_start(parlr)
phik_curr    = zeros(phik)
phik_curr_sum = zeros(phik)
cntr = 0
while cntr <= 10
    if cntr <= 3
        gradupdate!(phik_curr, datax, parlr)
    else
	    hmc!(phik_curr, datax, parlr)
	end
    phik_curr_sum += phik_curr
    cntr += 1
end

posterior_spectrum  = parlr.cNT .* parlr.cPP ./ (parlr.cNT .+ parlr.cPP) #
wfilter_datak  = posterior_spectrum .* fft2(datax, parlr) ./ parlr.cNT
errorsim_datak = white(parlr, d) .* sqrt(posterior_spectrum) 
wfilter_datax = ifft2r(wfilter_datak, parlr)
postsimx = ifft2r(wfilter_datak + errorsim_datak, parlr)
phix_curr = ifft2r(phik_curr, parlr)
phix = ifft2r(phik, parlr)

@imag4 wfilter_datax phix_curr phix postsimx
# @imag2 ifft2r(phik_curr, parlr) ifft2r(phik, parlr)








##
# test out the t-messanger gibbs algorithm
using CMB, FFT, INTERP
include("funcs.jl")
include("macros.jl")
parlr = setpar(pixel_size_arcmin=2.0, n=2.0^9,          beamFWHM=0.0, nugget_at_each_pixel=(5.0)^2, maskupL = 3000.0, masklowL = 1.0 )
parhr = setpar(pixel_size_arcmin=2.0./2.0, n=2.0*2.0^9, beamFWHM=0.0, nugget_at_each_pixel=(5.0)^2, maskupL = 3000.0, masklowL = 1.0 ) 
seed = rand(1:1000000)
srand(seed)
ytk_nomask, tildetk, phix, tx_hr = simulate_start(parlr)
phik = fft2(phix,parlr)
maskboolx =  (maximum(parlr.grd.x)*0.3) .<= parlr.grd.x .<= (maximum(parlr.grd.x)*0.45) 
maskvarx = parlr.nugget_at_each_pixel .* ones(size(phix))
maskvarx[maskboolx] = Inf
ytx = ifft2r(ytk_nomask, parlr)
ytx[maskboolx] = 0.0
ytk = fft2(ytx, parlr)
tx_hr_sim    = zeros(parhr.grd.x)
ttx_hr_sim    = zeros(parhr.grd.x)
tbarx_hr_sim    = zeros(parhr.grd.x)
phik_curr    = phik; "At the truth"


#gibbspass_cool2!(tx_hr_sim, ttx_hr_sim, phik_curr, ytx, maskvarx, parlr, parhr)
gibbspass_cool1!(tx_hr_sim, tbarx_hr_sim, phik_curr, ytx, maskvarx, parlr, parhr)

#@imag4 tx_hr_sim ttx_hr_sim tbarx_hr_sim ytx


#  look at the quadratic estiamte of this...
al = ttk_al(parlr)
phidx1_lr =  ifft2r(complex(0.0,1.0) .* parlr.grd.k1 .* phik_curr, parlr)
phidx2_lr =  ifft2r(complex(0.0,1.0) .* parlr.grd.k2 .* phik_curr, parlr)   
#tildesx_lr_curr = nearest_interp2(parhr.grd.x, parhr.grd.y, ttx_hr_sim, parlr.grd.x + phidx1_lr, parlr.grd.y + phidx2_lr) # this one has the uniform noise
tildetx_lr_curr =  spline_interp2(parhr.grd.x, parhr.grd.y, tx_hr_sim, parlr.grd.x + phidx1_lr, parlr.grd.y + phidx2_lr)
# sim_ytk = fft2(tildetx_lr_curr, parlr) + white(parlr).* sqrt(parlr.cNT)./(parlr.grd.deltk 
phik_curr = ttk_est(fft2(tildetx_lr_curr,parlr), parlr) .* (parlr.cPP ./ (2.0 .* al + parlr.cPP)) 

#------------------ save figures
fig = plt.figure(figsize=(12,12))
plt.subplot(2,2,3)
plt.imshow(phix, interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
plt.xlabel("true lensing potential")
plt.subplot(2,2,1)
plt.imshow(ifft2r(phik_curr,parlr), interpolation = "none", vmin=minimum(phix),vmax=maximum(phix)) 
plt.xlabel("estimated lensing potential")
plt.subplot(2,2,2)
plt.imshow(tx_hr_sim, interpolation = "none", vmin=minimum(tx_hr),vmax=maximum(tx_hr)) 
plt.xlabel("estimated unlensing CMB")
plt.subplot(2,2,4)
plt.imshow(ytx, interpolation = "none", vmin=minimum(tx_hr),vmax=maximum(tx_hr)) 
plt.xlabel("data")
# plt.savefig("$savepath/imag$bglp.png",dpi=100)
# plt.close(fig)
plt.show()








# for cntr = 1:100
#   #wsim_gibbs_d!(sx_lr_sim, sbarx_lr_sim, ytx, maskvarx, parlr, 1500, 10)
#   wsim_gibbs_t!(sx_lr_sim, tx_lr_sim, ytx, maskvarx, parlr, 10)
# end

# sk_lr_sim = fft2(sx_lr_sim, parlr).*(parlr.grd.r .<= 1500)
# @imag4 sbarx_lr_sim ifft2r(sk_lr_sim,parlr) tx_lr_sim ytx

# # wsim_gibbs_d!(sx_lr_sim, sbarx_lr_sim, ytx, maskvarx, parlr, 1500, 250)
# # @imag4 sbarx_lr_sim sx_lr_sim ytx ytx

# # wsim_gibbs_t!(sx_lr_sim, tx_lr_sim, ytx, maskvarx, parlr, 150)
# # @imag4 tx_lr_sim sx_lr_sim ytx ytx



