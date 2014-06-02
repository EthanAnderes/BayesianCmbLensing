

#------------------------------------
# HMC sampler
#--------------------------------------------
function hmc!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_hmc = 1.0e-3)
    ePs = scale_hmc * rand()
    ulim =  30 
    h0 = smooth_heavy(parlr.grd.r, 0.5, 1, 1500, 1/200) .* parlr.cPP ./ (parlr.grd.deltk^2) 
    mk = 1.0e-2 ./ h0
    mk[parlr.pMaskBool] = 0.0

    phik_test = copy(phik_curr)
    rk   = white(parlr) .* sqrt(mk); # note that the variance of real(pk_init) and imag(pk_init) is mk/2
    grad, loglike   = ttk_grad_wlog(tildetx_hr_curr, phik_test, parlr, parhr)
    h_at_zero = 0.5 * sum( abs2(rk[!parlr.pMaskBool])./(2*mk[!parlr.pMaskBool]/2)) - loglike # the 0.5 is out front since only half the sum is unique
    
    for HMCCounter = 1:ulim 
        loglike = lfrog!(phik_test, rk, tildetx_hr_curr, parlr, parhr, ePs, mk)
    end
    
    h_at_end = 0.5 * sum( abs2(rk[!parlr.pMaskBool])./(2*mk[!parlr.pMaskBool]/2)) - loglike # the 0.5 is out front since only half the sum is unique
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
function lfrog!(phik_curr, rk, tildetx_hr_curr, parlr, parhr, ePs, mk)
    grad, loglike   = ttk_grad_wlog(tildetx_hr_curr, phik_curr, parlr, parhr)
    rk_halfstep =  rk +  ePs .* grad ./ 2.0
    inv_mk = 1./ (mk ./ 2.0)
    inv_mk[parlr.pMaskBool] = 0.0
    phik_curr[:] = phik_curr + ePs .* inv_mk .* rk_halfstep
    grad, loglike   = ttk_grad_wlog(tildetx_hr_curr, phik_curr, parlr, parhr)
    rk[:] = rk_halfstep + ePs .* grad ./ 2.0
    loglike
end



#-----------------------------
# messenger algorithms: t messenger
#------------------------------------
function gibbspass_t!(sx, tx, phik_curr, ytx, maskvarx, parlr, parhr, coolingVec = [Inf for k=1:100])
    phidx1_hr_curr, phidx2_hr_curr, phidx1_lr_curr, phidx2_lr_curr = phi_easy_approx(phik_curr, parlr, parhr)
    dx, Nx = embedd(ytx, phidx1_lr_curr, phidx2_lr_curr, maskvarx, parlr, parhr)
    for uplim in coolingVec
        wsim_gibbs_t!(sx, tx, dx, Nx, parhr, uplim)
    end
    phidx1_hr_curr, phidx2_hr_curr, phidx1_lr_curr, phidx2_lr_curr
end
# ----- these are sub functions
function  wsim_gibbs_t!(sx, tx, dx, Nx, par, uplim)
    Tx = 0.99 * minimum(Nx)
    barNx = Nx .- Tx
    delt0 = 1.0./par.grd.deltk^2.0
    if uplim == Inf
        lamx = 1.0
    else
        lamx = (grd.deltx^2.0) .* par.CTell2d[round(uplim)]
        lamx = max(1.0, lamx)
    end
    sim_sx!(sx, tx, lamx * Tx, barNx, par)
    sim_tx!(sx, tx, dx, lamx * Tx, barNx, par)
end
function sim_sx!(sx, tx, Tx::Float64, barNx, par)
    tk  = fft2(tx, par)
    Tk  = Tx * par.grd.deltx * par.grd.deltx # this converts pixel space variance to spectrum
    pargrddeltk2 = par.grd.deltk * par.grd.deltk
    tmp    = 1.0./ (pargrddeltk2./par.cTT .+ pargrddeltk2./Tk) 
    sk_wf  = tmp .* tk ./ (Tk ./ pargrddeltk2)
    sk_flx = white(par) .* sqrt(tmp)
    sx[:]  = ifft2r(sk_wf + sk_flx, par)
end  
function sim_tx!(sx, tx, dx, Tx::Float64, barNx, par)
    tmp = 1.0./(1.0./barNx .+ 1.0./Tx)
    tx_wf  = tmp .* (dx./barNx + sx./Tx)
    tx_flx  = randn(size(tx)) .* sqrt(tmp) 
    tx[:] = tx_wf + tx_flx
end




##--------------------------------
# gradient of phi given  tildetx
#------------------------------------
function gradupdate!(phik_curr, tildetx_hr_curr, parlr, parhr, scale_grad=1.0e-3)
    for cntr = 1:30
        grad, loglike = ttk_grad_wlog(tildetx_hr_curr, phik_curr, parlr, parhr)
        println(loglike)
        phik_curr[:]  =  phik_curr + grad .* scale_grad .* smooth_heavy(parlr.grd.r, 1, 2, 1000, 1/200) .* parlr.cPP ./ (parlr.grd.deltk^2) 
    end
end
function ttk_grad_wlog(tildetx_hr_sim, phik_curr, parlr, parhr)
    parlrgrddeltk2 = parlr.grd.deltk * parlr.grd.deltk
    phidx1_lr =  ifft2r(complex(0.0,1.0) .* parlr.grd.k1 .* phik_curr, parlr)
    phidx2_lr =  ifft2r(complex(0.0,1.0) .* parlr.grd.k2 .* phik_curr, parlr)   
    tildetk_hr_sim   = fft2(tildetx_hr_sim, parhr)
    tildetxd1   = ifft2r(complex(0.0,1.0) .* parhr.grd.k1 .* tildetk_hr_sim, parhr) 
    tildetxd2   = ifft2r(complex(0.0,1.0) .* parhr.grd.k2 .* tildetk_hr_sim, parhr) 
    tildetx_unlensed     = spline_interp2(parhr.grd.x, parhr.grd.y, tildetx_hr_sim   , parlr.grd.x - phidx1_lr, parlr.grd.y - phidx2_lr, 0.01)
    tildetxd1_unlensed   = spline_interp2(parhr.grd.x, parhr.grd.y, tildetxd1  , parlr.grd.x - phidx1_lr, parlr.grd.y - phidx2_lr, 0.01)
    tildetxd2_unlensed   = spline_interp2(parhr.grd.x, parhr.grd.y, tildetxd2  , parlr.grd.x - phidx1_lr, parlr.grd.y - phidx2_lr, 0.01)
    tildetk_unlensed     =  fft2(tildetx_unlensed  , parlr);  tildetk_unlensed[parlr.cMaskBool]   = 0.0
    tildetkd1_unlensed   =  fft2(tildetxd1_unlensed, parlr);  tildetkd1_unlensed[parlr.cMaskBool] = 0.0
    tildetkd2_unlensed   =  fft2(tildetxd2_unlensed, parlr);  tildetkd2_unlensed[parlr.cMaskBool] = 0.0
    #--------------- log likelihood
    loglike = 0.0
    maskforpsi = one(phik_curr)
    maskforpsi[parlr.pMaskBool] = 0.0
    for k=2:length(phik_curr)
        loglike += - 0.25 * abs2(maskforpsi[k] * phik_curr[k]) / (parlr.cPP[k] / (2.0 * parlrgrddeltk2))
        loglike += - 0.25 * abs2(tildetk_unlensed[k]) / (parlr.cTT[k] / (2.0 * parlrgrddeltk2))
    end
    #--------------------------------------
    Bk = tildetk_unlensed ./ (parlr.cTT)
    Bk[parlr.cMaskBool] = 0.0
    Bx = ifft2r(Bk, parlr)
    constantterm =  complex(0.0,-2.0) * parlrgrddeltk2 # the minus is here to convert psi gradient to phi gradient
    term  = constantterm * parlr.grd.k1 .* fft2(ifft2r(tildetkd1_unlensed, parlr) .* Bx, parlr)
    term += constantterm * parlr.grd.k2 .* fft2(ifft2r(tildetkd2_unlensed, parlr) .* Bx, parlr)
    term -= 2.0 * parlrgrddeltk2 *  phik_curr ./ parlr.cPP
    term[parlr.pMaskBool] = 0.0
    term, loglike
end






#-----------------------------
# messenger algorithms: dual messenger
#------------------------------------
function gibbspass_d!(sx, sbarx, phik_curr, ytx, maskvarx, parlr, parhr, coolingVec = [Inf for k=1:100])
    phidx1_hr_curr, phidx2_hr_curr, phidx1_lr_curr, phidx2_lr_curr = phi_easy_approx(phik_curr, parlr, parhr)
    dx, Nx = embedd(ytx, phidx1_lr_curr, phidx2_lr_curr, maskvarx, parlr, parhr)
    for uplim in coolingVec
        wsim_gibbs_d!(sx, sbarx, dx, Nx, parhr, uplim)
    end
    sk = fft2(sx, parhr)
    sk[parhr.grd.r .> coolingVec[end]] = 0.0
    sx[:] = ifft2r(sk, parhr)
    phidx1_hr_curr, phidx2_hr_curr, phidx1_lr_curr, phidx2_lr_curr
end
# ---- these are sub functions
function  wsim_gibbs_d!(sx, sbarx, dx, Nx, par, uplim)
    Sbark, Qx, Qk = get_SQ(uplim, par)
    simd_sbarx!(sx, sbarx,  Sbark, Qk, par)
    simd_sx!(sx, sbarx, dx, Sbark, Qx, Nx, par)
end
function simd_sx!(sx, sbarx, dx, Sbark, Qx::Float64, Nx, par)
    tmp = 1.0 ./ (1.0./Nx .+ 1.0./Qx)
    sx_wf  = tmp .* (dx./Nx .+ sbarx./Qx)  
    sx_flx = randn(size(sx)) .* sqrt(tmp) 
    sx[:] = sx_wf + sx_flx
end  
function simd_sbarx!(sx, sbarx, Sbark, Qk::Float64, par)
    # tmp = 1.0 ./ (1.0./Qk + 1.0./Sbark) # this can work when Sbark is zeros by using the Wiener filter version 
    tmp = Qk .* Sbark ./ (Sbark .+ Qk) # this can work when Sbark is zero
    sbark_wf = tmp .* fft2(sx, par) ./ Qk
    sbark_flx = white(par) .* sqrt(tmp)
    sbark = sbark_wf + sbark_flx
    sbarx[:] = ifft2r(sbark, par)
end
function get_SQ(lp_init, par)
    delt0 = 1.0./par.grd.deltk^2.0
    lam = delt0 .* par.CTell2d[round(lp_init)]
    Sk = delt0 .* par.cTT # fourier variance
    Sbark = Sk .- lam 
    Sbark[Sbark .< 0.0]= 0.0
    Sbark[1] = 0.0
    Qk = lam    # so Qk + Sbark should be delt0.*par.cTT up to lp_init then lam after
                # Qk / delt0 is the spectrum
    Qx = Qk ./ (delt0 * par.grd.deltx^2.0) # to convert noise spectrum (i.e. Qk / delt0) to pixel varuance take spectrum / par.grd.deltx^2.0
    Sbark, Qx, Qk
end



#----------------------------------------------
# quadratic estimate
#----------------------------------------------
function ttk_est(ytk, par)
    cTT = par.cTT 
    cTTl_n_m = par.cTTLen + par.cNT .* par.cMaskInf
    cTTl_n = par.cTTLen + par.cNT
    tildetxd1_flt   = ifft2r(im * par.grd.k1 .* ytk .* cTT ./ cTTl_n_m, par) 
    tildetxd2_flt   = ifft2r(im * par.grd.k2 .* ytk .* cTT ./ cTTl_n_m, par)
    Bx_flt     = ifft2r(ytk ./ cTTl_n_m, par)
    ttk  = complex(0.0,-2.0) * par.grd.k1 .* fft2(tildetxd1_flt .* Bx_flt, par)
    ttk += complex(0.0,-2.0) * par.grd.k2 .* fft2(tildetxd2_flt .* Bx_flt, par)
    ttk = ttk .* ttk_al(par)
    ttk[par.pMaskBool] = 0.0
    ttk
end
function ttk_al(parin)
    cTT = parin.cTT 
    cTTl_n_m = parin.cTTLen + parin.cNT .* parin.cMaskInf
    cTTl_n = parin.cTTLen + parin.cNT 

    gttsqk = zero(parin.grd.k1)

    tmp  = 2*ifft2r(im .* parin.grd.k1 .* cTTl_n .* cTT ./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*parin.grd.k1.*cTTl_n.*(cTT)./(cTTl_n_m.*cTTl_n_m),parin)
    tmp += ifft2r(cTTl_n./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*parin.grd.k1.*im.*parin.grd.k1.*cTTl_n.*(cTT.*cTT)./(cTTl_n_m.*cTTl_n_m),parin)
    tmp += ifft2r(cTTl_n./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*parin.grd.k1.*im.*parin.grd.k1.*cTTl_n.*(cTT.*cTT)./(cTTl_n_m.*cTTl_n_m),parin)
    gttsqk += - 1.0 * parin.grd.k1 .* parin.grd.k1 .* (1./(2*pi)) .* fft2(tmp,parin) 
    
    tmp  = 2*ifft2r(im .* parin.grd.k2 .* cTTl_n .* cTT ./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*parin.grd.k1.*cTTl_n.*(cTT)./(cTTl_n_m.*cTTl_n_m),parin)
    tmp += ifft2r(cTTl_n./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*parin.grd.k2.*im.*parin.grd.k1.*cTTl_n.*(cTT.*cTT)./(cTTl_n_m.*cTTl_n_m),parin)
    tmp += ifft2r(cTTl_n./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*parin.grd.k2.*im.*parin.grd.k1.*cTTl_n.*(cTT.*cTT)./(cTTl_n_m.*cTTl_n_m),parin)
    gttsqk += - 2.0 * parin.grd.k2 .* parin.grd.k1 .* (1./(2*pi)) .* fft2(tmp,parin) 
    
    tmp  = 2*ifft2r(im .* parin.grd.k2 .* cTTl_n .* cTT ./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*parin.grd.k2.*cTTl_n.*(cTT)./(cTTl_n_m.*cTTl_n_m),parin)
    tmp += ifft2r(cTTl_n./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*parin.grd.k2.*im.*parin.grd.k2.*cTTl_n.*(cTT.*cTT)./(cTTl_n_m.*cTTl_n_m),parin)
    tmp += ifft2r(cTTl_n./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*parin.grd.k2.*im.*parin.grd.k2.*cTTl_n.*(cTT.*cTT)./(cTTl_n_m.*cTTl_n_m),parin)
    gttsqk += - 1.0 * parin.grd.k2 .* parin.grd.k2 .* (1./(2*pi)) .* fft2(tmp,parin) 
    
    1.0 ./ abs(real(gttsqk)) 
end
# this printed out the code that follows
# for ka = ["parin.grd.k1", "parin.grd.k2"], kbin = ["parin.grd.k1", "parin.grd.k2"]
    # println("""
        # tmp  = 2*ifft2r(im .* $ka .* cTTl_n .* cTT ./(cTTl_n_m.*cTTl_n_m),par) .*  ifft2r(im.*$kb.*cTTl_n.*(cTT)./(cTTl_n_m.*cTTl_n_m),par)
        # tmp += ifft2r(cTTl_n./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*$ka.*im.*$kb.*cTTl_n.*(cTT.*cTT)./(cTTl_n_m.*cTTl_n_m),parin)
        # tmp += ifft2r(cTTl_n./(cTTl_n_m.*cTTl_n_m),parin) .*  ifft2r(im.*$ka.*im.*$kb.*cTTl_n.*(cTT.*cTT)./(cTTl_n_m.*cTTl_n_m),parin)
        # gttsqk += - 1.0 * $ka .* $kb .* (1./(2*pi)) .* fft2(tmp,parin) 
        # """)
# end
function ttk_n0(parin) 
    tmp = 2.0 * ttk_al(parin)
    tmp[parin.cMaskBool] = Inf
    tmp
end




#----------------------------------
# simluation algorithm for generating synthetic data
#----------------------------------
function  simulate_start(par::SpectrumGrids; pad_proportion = 0.01)
    d=2.0
    #    simulate noise for Q and U  on low res
    znt =  ((par.grd.deltk/par.grd.deltx)^(d/2.0))*fft2(randn(size(par.grd.x)),par.grd.deltx)
    ntk = znt.*sqrt(par.cNT)./(par.grd.deltk^(d/2.0)) 
    #   simulate stationary Q,U on high reslution
    hr_factor = 3
    parhr= setpar(par.grd.pixel_size_arcmin/hr_factor, par.grd.n*hr_factor, par.beamFWHM, par.nugget_at_each_pixel, 3000.0, 2.0);
    zt   = ((parhr.grd.deltk/parhr.grd.deltx)^(d/2.0)) * fft2(randn(size(parhr.grd.x)),parhr.grd.deltx)
    tk   = zt.*sqrt(parhr.cTT)./(parhr.grd.deltk^(d/2.0)); 
    tx   = ifft2r(tk, parhr.grd.deltk)
    zP   = ((parhr.grd.deltk/parhr.grd.deltx)^(d/2.0)) * fft2(randn(size(parhr.grd.x)),parhr.grd.deltx) #the real and imag parts should have var 1/2.
    phik = zP.*sqrt(parhr.cPP)./(parhr.grd.deltk^(d/2.0))
    phix = ifft2r(phik, parhr.grd.deltk)
    phidx1 = ifft2r(complex(0.0,1.0) .* parhr.grd.k1 .* phik, parhr)
    phidx2 = ifft2r(complex(0.0,1.0) .* parhr.grd.k2 .* phik, parhr)    
    #   Lense Q,U
    tildetx = spline_interp2(parhr.grd.x, parhr.grd.y, tx, parhr.grd.x + phidx1, parhr.grd.y + phidx2, pad_proportion)
    #tildetx = ifft2r(parhr.beam.*fft2(tildetx,parhr.grd.deltx), parhr.grd.deltk) #h the beam is already a part of the noise...
    tildetk = fft2(tildetx[1:hr_factor:end, 1:hr_factor:end], par)
    ytk = ntk + tildetk
    ytk, tildetk, phix[1:hr_factor:end, 1:hr_factor:end], tx
end


#----------------------------------
# miscilanious function
#----------------------------------
white(par::SpectrumGrids) = (par.grd.deltk/par.grd.deltx)*fft2(randn(size(par.grd.x)),par.grd.deltx)
function upsample(psik_lr::Array{Complex{Float64},2} ,parlr, parhr)
    index =  (-parlr.grd.nyq-parlr.grd.deltk./10) .<= parhr.grd.kside .< (parlr.grd.nyq-parlr.grd.deltk./10)
    psik_hr = zeros(Complex{Float64},size(parhr.grd.k1))
    psik_hr[index,index]=psik_lr
    return psik_hr
end
function upsample(txlr::Array{Float64,2} ,parlr, parhr)
    tklr = fft2(txlr, parlr)
    index =  (-parlr.grd.nyq-parlr.grd.deltk./10) .<= parhr.grd.kside .< (parlr.grd.nyq-parlr.grd.deltk./10)
    tkhr = zeros(Complex{Float64},size(parhr.grd.k1))
    tkhr[index,index]=tklr
    ifft2r(tkhr, parhr)
end
function downsample(txhr::Array{Float64,2} ,parlr, parhr)
    freq = round(parlr.grd.deltx / parhr.grd.deltx)
    txhr[1:freq:end,1:freq:end]
end
function  phi_easy_approx(phik, parlr, parhr)
    phik_hr = upsample(phik, parlr, parhr)
    phidx1_hr = ifft2r(complex(0.0,1.0) .* parhr.grd.k1 .* phik_hr, parhr)
    phidx2_hr = ifft2r(complex(0.0,1.0) .* parhr.grd.k2 .* phik_hr, parhr)
    phidx1_lr = ifft2r(complex(0.0,1.0) .* parlr.grd.k1 .* phik, parlr)
    phidx2_lr = ifft2r(complex(0.0,1.0) .* parlr.grd.k2 .* phik, parlr)
    phidx1_hr, phidx2_hr, phidx1_lr, phidx2_lr
end 
function embedd(ytx, phidx1_lr, phidx2_lr, Nx_lowres, parlr, parhr; pad_proportion = 0.1)
    index_matrix1 = reshape(1:length(parhr.grd.x),size(parhr.grd.x)); #this index matrix only needs to be made once
    I_obs_into_xyfull = int(nearest_interp2(parhr.grd.x ,parhr.grd.y, index_matrix1, parlr.grd.x + phidx1_lr, parlr.grd.y + phidx2_lr; pad_proportion = pad_proportion))
    dx = zeros(Float64,size(parhr.grd.x))
    dx[I_obs_into_xyfull]  = vec(ytx)
    NN = fill(Inf, size(dx))
    NN[I_obs_into_xyfull]= vec(Nx_lowres)
    dx, NN
end
function smooth_heavy(x,height1,height2,location, transition_sharpness)
    zeroto1 = 0.5 .+ 0.5 .* tanh(transition_sharpness .* (x.-location))
    (height2 .- height1) .* zeroto1 .+ height1
end
function radialAve(Xk,par)
    block_counter = 0.0
    block_delta = par.grd.deltk
    radXk = zero(Xk)
    while true
      k_index   = ((block_counter)*block_delta .< par.grd.r .<= (block_counter + 3)*block_delta) & (1 .< par.grd.r) 
      all(~k_index) && break
      radXk[k_index] =  mean( Xk[k_index] ) 
      block_counter += 1.0 
    end
    radXk
end
#  plt.semilogy(parglob.grd.r[:,1], radialAve(abs2(ttk1),parglob)[:,1]); plt.show()
#  plt.semilogy(parglob.grd.r[:,1], radialAve(ttk_al(parglob),parglob)[:,1]); plt.show()


function pos(x) 
        y = real(x)
        y[y.<0.0] = 0.0
        y
end 



