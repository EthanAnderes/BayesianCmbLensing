using PyCall, Interp
@pyimport scipy.interpolate as scii


##  pixel grid and Fourier conjugate grid in radians
immutable Grid_xandk
    pixel_size_arcmin::Float64
    n::Float64
    deltx::Float64
    deltk::Float64
    period::Float64
    nyq::Float64
    kside::Array{Float64,1}
    x::Array{Float64,2}
    y::Array{Float64,2}
    k1::Array{Float64,2}
    k2::Array{Float64,2}
    r::Array{Float64,2}
end
function Grid_xandk(pixel_size_arcmin, n) 
    deltx = pixel_size_arcmin * pi / (180 * 60) #this is in radians
    x,y = meshgrid(1:n, 1:n)
    x .*= deltx
    x .-= minimum(x)
    y .*= deltx
    y .-= minimum(y)
    kside  = (2 * pi / deltx) .* (mod(1/2 .+ [0:(n-1)]./n, 1) .- 1/2)
    k1,k2  = meshgrid(kside, kside)
    period = deltx * n
    deltk  = 2 * pi / period  
    nyq    = 2 * pi / (2 * deltx)
    r      = sqrt(k1.^2 .+ k2.^2)
    return Grid_xandk(pixel_size_arcmin, n, deltx, deltk, period, nyq, kside, x, y, k1, k2, r) 
end



immutable SpectrumGrids # this contains all the signal + noise spectral information
    grd::Grid_xandk # everything is defined in reference to a grid
    ell::Array{Float64,1}
    ellLen::Array{Float64,1}
    #------------------------
    cMaskBool::BitArray{2} # this gives l_max for cmb
    pMaskBool::BitArray{2} # this gives l_max for phi
    cMaskInf::Array{Float64,2}
    #---------- noise
    nugget_at_each_pixel::Float64
    beamFWHM::Float64
    beam::Array{Float64,2}
    # radial profile
    CTell2d::Array{Float64,1}
    CTell2dLen::Array{Float64,1}
    CPell2d::Array{Float64,1}
    # matrices
    cTT::Array{Float64,2}
    cTTLen::Array{Float64,2}
    cPP::Array{Float64,2}
    cNT::Array{Float64,2}    
end
function SpectrumGrids(
    ell, 
    ellLen, 
    CPell2d, 
    CTell2d, 
    CTell2dLen,  
    pixel_size_arcmin, 
    n, 
    beamFWHM, 
    nugget_at_each_pixel, 
    maskupC, 
    maskupP
)
    d = 2.0
    grd = Grid_xandk(pixel_size_arcmin, n)
    sig=(beamFWHM * (pi / (180 * 60))) / ( 2 * sqrt(2 * log(2)))
    beam = exp(-(sig^2) .* (grd.k1.^2 .+ grd.k2.^2) ./ 2)
    beamSQ =abs2(beam)
    cNT    = nugget_at_each_pixel .* (grd.deltx^d) ./ beamSQ

    cMaskBool = ~((grd.r.<= maskupC) & (grd.r.>= 1.0)) # this is true when the masking is active Mat[cMaskBool] = 0 or Inf
    pMaskBool = ~((grd.r.<= maskupP) & (grd.r.>= 1.0)) # this is true when the masking is active Mat[cMaskBool] = 0 or Inf
    cMaskInf = 1.0./((grd.r.<=maskupC) & (grd.r.>= 1.0))  # this equals Inf when the masking is active and 1.0 otherwise...
    
    index=ceil(grd.r)
    index[find(index.==0)]=1

    logCPP = linear_interp1(ell,log(CPell2d), index)
    logCPP[find(logCPP .== 0)] =-Inf
    logCPP[find(isnan(logCPP))]=-Inf
    cPP = exp(logCPP)
    logCTT = linear_interp1(ell,log(CTell2d), index)
    logCTT[find(logCTT .== 0)] =-Inf
    logCTT[find(isnan(logCTT))]=-Inf
    cTT = exp(logCTT)
    logCTTLen = linear_interp1(ellLen,log(CTell2dLen), index)
    logCTTLen[find(logCTTLen .== 0)] =-Inf
    logCTTLen[find(isnan(logCTTLen))]=-Inf
    logCTTLen[grd.r .> 8000]=-Inf
    cTTLen = exp(logCTTLen)

    inputs = {grd,
        ell,
        ellLen,
        cMaskBool,
        pMaskBool,
        cMaskInf,
        nugget_at_each_pixel,
        beamFWHM,
        beam,
        CTell2d,
        CTell2dLen,
        CPell2d,
        cTT,
        cTTLen,
        cPP,
        cNT }
    SpectrumGrids(inputs...)
end




# get spectra and an instance of SpectrumGrids
function setpar(pixel_size_arcmin, n, beamFWHM, nugget_at_each_pixel, maskupC, maskupP)
    Sfile = open("src/camb/test_scalCls.dat")
    lines = readlines(Sfile)
    close(Sfile)
    CLs = Array(Any,(length(lines),6))
    for i in 1:length(lines)
        CLs[i,:] = lines[i] |> strip |> chomp |> split |> transpose
    end
    CLs = map(parsefloat, CLs)
    ell=[1;CLs[:,1]]
    CTell2d=[0;CLs[:,2].*(2*pi)./(CLs[:,1].*(CLs[:,1] .+ 1))]
    CPell2d=[0;CLs[:,5]./(7.4311e12*CLs[:,1].^4)]

    LenSfile = open("src/camb/test_lensedCls.dat")
    Lenlines = readlines(LenSfile)
    close(LenSfile)
    LenCLs = Array(Any,(length(Lenlines),5))
    for i in 1:length(Lenlines)
        LenCLs[i,:] = Lenlines[i] |> strip |> chomp |> split |> transpose
    end
    LenCLs = map(parsefloat, LenCLs)  #[ell, CTT, CEE, CBB, CTE]
    ellLen = [1;LenCLs[:,1]]
    CTell2dLen =[0;LenCLs[:,2].*(2*pi)./(LenCLs[:,1].*(LenCLs[:,1].+1))]
   
   inputs = {ell, 
        ellLen, 
        CPell2d, 
        CTell2d, 
        CTell2dLen,  
        pixel_size_arcmin, 
        n, 
        beamFWHM, 
        nugget_at_each_pixel, 
        maskupC, 
        maskupP}
    SpectrumGrids(inputs...)
end


#  if you need to specify where src is...used in graphics
function setpar(pixel_size_arcmin, n, beamFWHM, nugget_at_each_pixel, maskupC, maskupP, pathtosrc::String)
    Sfile = open(joinpath(pathtosrc,"camb/test_scalCls.dat"))
    lines = readlines(Sfile)
    close(Sfile)
    CLs = Array(Any,(length(lines),6))
    for i in 1:length(lines)
        CLs[i,:] = lines[i] |> strip |> chomp |> split |> transpose
    end
    CLs = map(parsefloat, CLs)
    ell=[1;CLs[:,1]]
    CTell2d=[0;CLs[:,2].*(2*pi)./(CLs[:,1].*(CLs[:,1] .+ 1))]
    CPell2d=[0;CLs[:,5]./(7.4311e12*CLs[:,1].^4)]

    LenSfile = open(joinpath(pathtosrc,"camb/test_lensedCls.dat"))
    Lenlines = readlines(LenSfile)
    close(LenSfile)
    LenCLs = Array(Any,(length(Lenlines),5))
    for i in 1:length(Lenlines)
        LenCLs[i,:] = Lenlines[i] |> strip |> chomp |> split |> transpose
    end
    LenCLs = map(parsefloat, LenCLs)  #[ell, CTT, CEE, CBB, CTE]
    ellLen = [1;LenCLs[:,1]]
    CTell2dLen =[0;LenCLs[:,2].*(2*pi)./(LenCLs[:,1].*(LenCLs[:,1].+1))]
   
   inputs = {ell, 
        ellLen, 
        CPell2d, 
        CTell2d, 
        CTell2dLen,  
        pixel_size_arcmin, 
        n, 
        beamFWHM, 
        nugget_at_each_pixel, 
        maskupC, 
        maskupP}
    SpectrumGrids(inputs...)
end


