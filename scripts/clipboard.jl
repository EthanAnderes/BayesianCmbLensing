#######################
#
# test out a new messenger algorithm
#
#####################
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
tx = tx_hr[1:3:end, 1:3:end]
tk = fft2(tx, parlr)

tmpdo = maximum(parlr.grd.x)*0.3
tmpup = maximum(parlr.grd.x)*0.4
maskboolx = tmpdo .<= parlr.grd.x .<= tmpup
maskvarx  = parlr.nugget_at_each_pixel .* ones(size(parlr.grd.x))
maskvarx[maskboolx] = Inf


function makeAmults(maskboolx, par)
  function A_mult(sk)
    maskAsx = ifft2r(sk, par)
    maskAsx[maskboolx] = 0.0
    maskAsx
  end
  function Astar_mult(zx)
    maskzx = zeros(zx)
    maskzx[!maskboolx] = zx[!maskboolx]
    fft2(maskzx, par)
  end
  A_mult, Astar_mult
end



function makeAmultsv2(maskboolx, par)
  # this one has the spectral multiplier in here too
  delt0 = 1 / par.grd.deltk ^ 2
  function A_mult(sk)
    maskAsx = ifft2r(√(delt0 * par.cTT) .* sk, par)
    maskAsx[maskboolx] = 0.0
    maskAsx
  end
  function Astar_mult(zx)
    maskzx = zeros(zx)
    maskzx[!maskboolx] = zx[!maskboolx]
    tmpk = fft2(maskzx, par) .* √(delt0 * par.cTT)
    tmpk[1] = complex(0.0)
    tmpk
  end
  A_mult, Astar_mult
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
    fpwr[i] = sum(abs2(fk[ibin])) / (length(fk[ibin]) - 2)
  end
  # --- now interpolate on to kmag
  fpwr_matrix = zero(kmag)
  for i in 1:size(kmag, 1), j in 1:size(kmag, 2)
    fpwr_matrix[i, j] = fpwr[indmin(abs(kmag[i,j] - bin_mids))] 
  end
  fpwr_matrix
end


#=
A_mult, Astar_mult = makeAmults(maskboolx, parlr)
σ² = 10.0
dx = A_mult(tk) + √(σ²) * randn(size(tildetk))
sk = zero(tk)
zx = zero(tx)
δ = sum(!maskboolx)/length(maskboolx)
function amp_pass!(sk, zx, dx, σ², δ, λᵗ, par)
  Δk     = par.grd.deltk * par.grd.deltk
  Δx     = par.grd.deltx * par.grd.deltx
  delt0  = 1 / Δk  
  #σ² * Δx * delt0 # start lambda at the fourier noise variance
  Sk     = delt0 * par.cTT
  for cntr in 1:250
    # --- update s
    exps = Sk ./ (λᵗ + Sk)
    exps[1] = 0.0
    sk[:]  = exps .* (sk + Astar_mult(zx))
    # --- update z
    # zx[:]  = dx - A_mult(sk) +  zx .* mean(exps) / δ #!!!!! I don't understand this last term?? why does it
                                                    # in particular why does it make sk + Astar_mult(zx) behave 
                                                    # like truth plus noise.
                                                    # is it possible to read the proof directly then reverse engerner
                                                    # what it should be
    zx[:]  = dx - A_mult(sk) + sum(exps) / δ # ./ (2par.grd.r + 1)        
    # --- update λᵗ
    # vars = λᵗ .* Sk ./ (λᵗ + Sk)
    # λᵗ[:] =  pos( binpower(sk + Astar_mult(zx), par.grd.r, (2 * par.grd.deltk):(2 * par.grd.deltk):5_000 ) - Sk)
    λᵗ = mean(abs2(zx)) / δ
  end
  λᵗ
end
λᵗ = 1e-5 * ones(size(sk)) 
λᵗ = amp_pass!(sk, zx, dx, σ², δ, λᵗ, parlr)

=#

function smooth(zx, par)
  zk = fft2(zx, par) 
  zk[par.grd.r .≥ 200.0] = 0.0
  ifft2r(zk, par)
end



A_mult, Astar_mult = makeAmultsv2(maskboolx, parlr)
delt0 =  1 / parlr.grd.deltk / parlr.grd.deltk
σ² = 10.0
tmpk = tk ./ √(parlr.cTT * delt0)
tmpk[1] = complex(0.0)
dx = A_mult(tmpk) + √(σ²) * randn(size(tildetk))
sk = zero(tk)
zx = zero(tx)
δ = sum(!maskboolx)/length(maskboolx)
function amp_passv2!(sk, zx, dx, σ², δ, λᵗ, par)
  Δk     = par.grd.deltk * par.grd.deltk
  Δx     = par.grd.deltx * par.grd.deltx
  delt0  = 1 / Δk  
  for cntr in 1:50
    # --- update s
    exps = λᵗ
    sk[:]  = exps .* (sk + Astar_mult(zx))
    # --- update z
    zx[:]  = dx - A_mult(sk)  #+ exps # 
    # --- update λᵗ
    # λᵗ = mean(abs2(zx)) 
    # λᵗ = mean(abs2(sk + Astar_mult(zx))) 
    λᵗ = smooth(abs2(sk + Astar_mult(zx)), par)
  end
  λᵗ
end
λᵗ = 1.0
λᵗ = amp_passv2!(sk, zx, dx, σ², δ, λᵗ, parlr)
subplot(1,2,1)
imshow(ifft2r(√(delt0 * parlr.cTT) .* sk, parlr))
subplot(1,2,2)
imshow(tx)


figure(figsize = (14,5))
subplot(1,2,1)
imshow(ifft2r(√(delt0 * par.cTT) .* sk, parlr))
subplot(1,2,2)
plot(λᵗ[1:10,1])
plot(parlr.cTT[1:10,1] / parlr.grd.deltk / parlr.grd.deltk)
#imshow( abs2(sk + Astar_mult(zx))- delt0 * par.cTT)



dt = ifft2r(sk, parlr)-tx
dd = dx-tx

plot(dt[:,1])
plot(dd[:,1])


plot(tx[:,1])
plot(ifft2r(sk, parlr)[:,1])
plot(dx[:,1])







# profile hmc etc to speed things up
const percentNyqForC = 0.5 # used for T l_max
const numofparsForP  = 1500  # used for P l_max
const hrfactor = 2.0
const pixel_size_arcmin = 2.0
const n = 2.0^9
const beamFWHM = 0.0
const nugget_at_each_pixel = (4.0)^2
begin  #< ---- dependent run parameters
  local deltx =  pixel_size_arcmin * pi / (180 * 60) #rads
  local period = deltx * n # side length in rads
  local deltk =  2 * pi / period
  local nyq = (2 * pi) / (2 * deltx)
  const maskupP  = √(deltk^2 * numofparsForP / pi)  #l_max for for phi
  const maskupC  = min(9000.0, percentNyqForC * (2 * pi) / (2 * pixel_size_arcmin * pi / (180*60))) #l_max for for phi
end
const scale_grad =  2.0e-3
const scale_hmc  =  0.8e-3

# ------------ load modules and functions
push!(LOAD_PATH, pwd()*"/src")
using Interp
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
tmpdo = maximum(parlr.grd.x)*0.3
tmpup = maximum(parlr.grd.x)*0.4
maskboolx = tmpdo .<= parlr.grd.x .<= tmpup
maskvarx  = parlr.nugget_at_each_pixel .* ones(size(parlr.grd.x))
maskvarx[maskboolx] = Inf
ytx = ifft2r(ytk_nomask, parlr)
ytx[maskboolx] = 0.0
ytk = fft2(ytx, parlr)

acceptclk   = [1] #initialize acceptance record
tx_hr_curr  = zero(parhr.grd.x)
ttx         = zero(parhr.grd.x)
p1hr, p2hr  = zero(parhr.grd.x), zero(parhr.grd.x)
phik_curr   = zero(fft2(ytx, parlr))
tildetx_hr_curr = zero(parhr.grd.x) 

@time grad, loglike   = ttk_grad_wlog(tildetx_hr_curr, phik_curr, parlr, parhr)
@time grad, loglike   = ttk_grad_wlog(tildetx_hr_curr, phik_curr, parlr, parhr);
# lfrog!(phik_curr, phik_curr, tildetx_hr_curr, parlr, parhr, 0.0001, phik_curr)
# hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc)

Profile.clear()  # in case we have any previous profiling data
@profile   grad, loglike   = ttk_grad_wlog(tildetx_hr_curr, phik_curr, parlr, parhr)
# @profile lfrog!(phik_curr, phik_curr, tildetx_hr_curr, parlr, parhr, 0.0001, phik_curr)
# @profile  hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc)
using ProfileView
ProfileView.view()
















#######################################
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
using Interp
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
ttx         = zero(parhr.grd.x)
p1hr, p2hr  = zero(parhr.grd.x), zero(parhr.grd.x)
phik_curr   = zero(fft2(ytx, parlr))
tildetx_hr_curr = zero(parhr.grd.x) 



function  rft(fx, par::SpectrumGrids)
  c = complex( (par.grd.deltx / √(2.0 * π))^2.0 )
  fk = rfft(fx)
  scale!(fk, c)
  fk
end
function  irft(fk, par::SpectrumGrids)
  c = (par.grd.deltk / √(2.0 * π))^2.0 
  nint = int(par.grd.n)
  fx = brfft(fk, nint)
  scale!(fx, c)
  fx
end
rwhite(ln, par::SpectrumGrids) = (par.grd.deltk/par.grd.deltx)*rft(randn(ln, ln), par)
function gibbspass_t!(sx, tx, phik_curr, ytx, maskvarx, parlr, parhr, coolingVec = [Inf for k=1:100])
    phidx1_hr_curr, phidx2_hr_curr, phidx1_lr_curr, phidx2_lr_curr = phi_easy_approx(phik_curr, parlr, parhr)
    dx, Nx = embedd(ytx, phidx1_lr_curr, phidx2_lr_curr, maskvarx, parlr, parhr)
    tk    = rft(tx, parhr)
    sk    = similar(tk)
    gpass!(sx, sk, tx, tk, dx, Nx, parhr, coolingVec)
    phidx1_hr_curr, phidx2_hr_curr
end
function gpass!(sx, sk, tx, tk, dx, Nx, par, cool)
  ln    = size(tx, 1)
  d2k   = par.grd.deltk * par.grd.deltk
  d2x   = par.grd.deltx * par.grd.deltx
  delt0 = 1 / d2k
  barNx = 0.99 * minimum(Nx)
  barNk = barNx * d2x * delt0 # var in fourier
  tldNx = Nx .- barNx 
  Sk    = scale(par.cTT[1:size(sk, 1),1:size(sk, 2)], delt0)
  for uplim in cool
    λbarNk = (uplim > 8000.0) ? barNk : max(barNk, delt0 * par.CTell2d[int(uplim)])  # var in fourier
    λbarNx = λbarNk / (d2x * delt0)
    # --- update s
    tmpx   = 1 ./ (1 / λbarNk .+ 1 ./ Sk) 
    sk[:]  = tmpx .* scale(tk, 1 / λbarNk) 
    sk[:] += vec(rwhite(ln, par) .* √(tmpx))  
    sx[:]  = irft(sk, par) 
    # --- update t
    tmpx  = 1 ./ (1 ./ tldNx .+ 1 / λbarNx) 
    tx[:]   = tmpx .* (dx ./ tldNx + sx ./ λbarNx) 
    tx[:]  += vec(randn(ln, ln) .* √(tmpx))        
    tk[:]   = rft(tx, par)
  end
  nothing
end




cool =  linspace(100, 3000, 5)
@time p1hr[:], p2hr[:] = gibbspass_t!(tx_hr_curr, ttx, phik_curr, ytx, 
      maskvarx, parlr, parhr, cool
      );
@time p1hr[:], p2hr[:] = gibbspass_t_test!(tx_hr_curr, ttx, phik_curr, ytx, 
      maskvarx, parlr, parhr, cool
      ); 



srand(1)
tx_hr_curr1  = zero(parhr.grd.x)
ttx1         = zero(parhr.grd.x)
 p1hr[:], p2hr[:] = gibbspass_t_test!(tx_hr_curr1, ttx1, phik_curr, ytx, 
      maskvarx, parlr, parhr, cool
      );


srand(1)
tx_hr_curr2  = zero(parhr.grd.x)
ttx2         = zero(parhr.grd.x)
 p1hr[:], p2hr[:] = gibbspass_t!(tx_hr_curr2, ttx2, phik_curr, ytx, 
      maskvarx, parlr, parhr, cool
      );


plt.subplot(2,2,1)
plt.imshow(ytx);
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(tx_hr_curr2);
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(tx_hr_curr1);
plt.colorbar()
 plt.show()









# profile view this...
sx = tx_hr_curr
tx = ttx
phidx1_hr_curr, phidx2_hr_curr, phidx1_lr_curr, phidx2_lr_curr = phi_easy_approx(phik_curr, parlr, parhr)
dx, Nx = embedd(ytx, phidx1_lr_curr, phidx2_lr_curr, maskvarx, parlr, parhr)
tk    = rft(tx, parhr)
sk    = similar(tk)

Profile.clear()  # in case we have any previous profiling data
@profile  gpass!(sx, sk, tx, tk, dx, Nx, parhr, cool)
using ProfileView
ProfileView.view()





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



