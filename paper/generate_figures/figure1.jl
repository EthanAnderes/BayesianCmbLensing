"""
Generate figure 1 which shows the two parameter gibbs toy model
===================================================== """

using PyPlot
srcpath  = "/Users/ethananderes/Dropbox/BayesLense/src/"
savepath = "/Users/ethananderes/Google\ Drive/BayesLenseRev1/paper_rev2/"
seed = 1
srand(seed)


"""
Set up the Gibbs chain
------------------------------------------"""
rho = -0.99
sigma1 = [1 rho; rho 1]
sigmainv1 = inv(sigma1)
sigmainv2 = [1 0;-1 1]* sigmainv1 * [1 -1;0 1]
sigma2 = inv(sigmainv2)
pdfimag1(x) = (1/ (2 * pi)) * sqrt(det(sigmainv1)) * exp(-0.5 * (transpose(x) * sigmainv1 * x)[1]) 
pdfimag2(x) = (1/ (2 * pi)) * sqrt(det(sigmainv2)) * exp(-0.5 * (transpose(x) * sigmainv2 * x)[1])

xlim = 2.0
space = 0.01
gridpoints = Array{Float64,1}[[x,y] for x=(-xlim):space:xlim, y=(-xlim):space:xlim] #|> transpose |> flipud
imag1 = map(pdfimag1, gridpoints)
imag2 = map(pdfimag2, gridpoints)

sum(imag1 * ^(space,2))
function findContours(density, gridarea, probvec)
	hts = Float64[]
	prb = Float64[]
	for dh in density[1:50:end]
		push!(hts, dh)
		push!(prb, sum( gridarea*density[density .> dh] ))
	end
	answr = Float64[]
	for k in probvec
		toss, index = findmin(abs(prb .- k))
		push!(answr, hts[index])
	end
	answr
end



"""
Run the chain for 20 steps
------------------------------------------"""
thetachain1 = [0.0]
phichain1 = [0.0]
for k = 1:20
  push!(thetachain1, sigma1[1,2] * phichain1[end] * sqrt(sigma1[1,1] / sigma1[2,2]) + randn() * sqrt(1-sigma1[1,2]^2) * sqrt(sigma1[1,1]) )
  push!(phichain1, phichain1[end])
  push!(phichain1,  sigma1[1,2] * thetachain1[end] * sqrt(sigma1[2,2] / sigma1[1,1]) + randn() * sqrt(1-sigma1[1,2]^2) * sqrt(sigma1[2,2]) )
  push!(thetachain1, thetachain1[end])
end 

tildethetachain2 = [0.0]
phichain2 = [0.0]
for k = 1:20
  push!(tildethetachain2, sigma2[1,2] * phichain2[end] * sqrt(sigma2[1,1] / sigma2[2,2]) + randn() * sqrt(1-sigma2[1,2]^2) * sqrt(sigma2[1,1]) )
  push!(phichain2, phichain2[end])
  push!(phichain2,  sigma2[1,2] * tildethetachain2[end] * sqrt(sigma2[2,2] / sigma2[1,1]) + randn() * sqrt(1-sigma2[1,2]^2) * sqrt(sigma2[2,2]) )
  push!(tildethetachain2, tildethetachain2[end])
end 

probvec = [.5, .85]
cnt1 =  findContours(imag1, space^2, probvec)
cnt2 =  findContours(imag2, space^2, probvec)




"""
figure1a.pdf
------------------------------------------"""
contour(imag1, 
	levels = cnt1, 
	origin= "lower", 
	extent=[-2.5,2.5,-2.5,2.5], 
	colors="k", 
	linewidth=1.5, 
	label=L"$P(t, \varphi|data)$"
	)
plot(phichain1,thetachain1, linewidth=1.5)
ax = gca()
ax[:spines]["right"][:set_color]("none")
ax[:spines]["top"][:set_color]("none")
ax[:xaxis][:set_ticks_position]("bottom")
ax[:spines]["bottom"][:set_position](("data",0))
ax[:yaxis][:set_ticks_position]("left")
ax[:spines]["left"][:set_position](("data",0))
ax[:text](0.6, 0.6, L"$P$ $(t, \varphi|data)$", fontsize=27)
xticks([-2,-1,1,2], fontsize = 15)
yticks([-2,-1,1,2], fontsize = 15)
savefig(joinpath(savepath, "figure1a.pdf"), dpi=300, bbox_inches="tight", pad_inches=0) 



"""
figure1b.pdf
------------------------------------------"""
contour(imag2, levels = cnt2, origin= "lower", extent=[-2.5,2.5,-2.5,2.5], colors="k", linewidth=1.5)
xlabel(L"$\varphi$", fontsize = 28),
ylabel(L"$\widetilde{t}$", fontsize = 28)
plot(phichain2, tildethetachain2, linewidth=1.5)
ax = gca()
ax[:spines]["right"][:set_color]("none")
ax[:spines]["top"][:set_color]("none")
ax[:xaxis][:set_ticks_position]("bottom")
ax[:spines]["bottom"][:set_position](("data",0))
ax[:yaxis][:set_ticks_position]("left")
ax[:spines]["left"][:set_position](("data",0))
ax[:text](0.7, 0.7, L"$P$ $(\widetilde{t}, \varphi|data)$", fontsize=27)
xticks([-2,-1,1,2], fontsize = 15)
yticks([-2,-1,1,2], fontsize = 15)
savefig(joinpath(savepath, "figure1b.pdf"), dpi=300, bbox_inches="tight", pad_inches=0) 


