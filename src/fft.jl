
function  fft2(fx, deltx::Real)
	d = 2.0
	c = complex( (deltx / √(2.0 * π))^d )
   fk = fft(fx)
   scale!(fk, c)
   fk
end
function  ifft2(fk, deltk::Real)
	d = 2.0
	c = (deltk / √(2.0 * π))^d
	fx = bfft(fk)
	scale!(fx, c)
	fx
end
function  ifft2r(fk, deltk::Real)
	d = 2.0
	c = (deltk / √(2.0 * π))^d
	fx = real(bfft(fk))
	scale!(fx, c)
	fx
end
fft2(fx, par::SpectrumGrids) = fft2(fx, par.grd.deltx)
ifft2(fk, par::SpectrumGrids) = ifft2(fk, par.grd.deltk)
ifft2r(fk, par::SpectrumGrids) = ifft2r(fk, par.grd.deltk)
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

