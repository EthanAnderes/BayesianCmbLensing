


#-------------------
# fft2
#----------------------------
function  fft2(xx, deltx::Real)
	#2-d Fourier transform of Xx to Xk
	#returns Xk=FFT2(Xx, Deltx)
	d = 2.0
	c = (deltx/sqrt(2.0*pi))^d 
  	scale(fft( xx ), c)
end
function  fft2(xx, deltx::Real, plan)
	d=2.0
	c= (deltx/sqrt(2.0*pi))^d 
  	scale(plan( xx ), c)
end
function fft2(xx ,par::SpectrumGrids)
	if size(xx)==size(par.grd.k1)
		return fft2(xx, par.grd.deltx)
	else
		return fft2(xx, par.grd.deltx)
	end
end

#-------------------
# ifft2r
#----------------------------
#TODO extend this so that it is the inverse of fft2r...
#the main issue is that rfft returns a vector which has the first dimension cut in half...
function  ifft2r(xk, deltk::Real)
	#inverse 2-d Fourier transform when Xx is real
	# Xx = IFFT2r(Xk, Deltk)
	d=2.0
	n=float64(size(xk,1))
	c= (n*deltk/sqrt(2.0*pi))^d 
	scale(real(ifft( xk )), c)
end
function  ifft2r(xk, deltk::Real, plan)
	#inverse 2-d Fourier transform when Xx is real
	# Xx = IFFT2r(Xk, Deltk)
	d=2.0
	n=float64(size(xk,1))
	c= (n*deltk/sqrt(2.0*pi))^d 
	scale(real(plan( xk )), c)
end
function ifft2r(xk, par::SpectrumGrids)
	if size(xk)==size(par.grd.k1)
		return ifft2r(xk, par.grd.deltk)
	else
		return ifft2r(xk, par.grd.deltk)
	end
end



#-------------------
# ifft2r
#----------------------------
function  ifft2(xk, deltk::Real)
	#inverse 2-d Fourier transform when Xx could be imag
	# Xx = IFFT2r(Xk, Deltk)
	d=2.0
	n=float64(size(xk,1))
	c=(n*deltk/sqrt(2.0*pi))^d
	scale( ifft( xk ), c)
end

 



