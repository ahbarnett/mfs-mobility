# test MFS3D

include("MFS3D.jl")   # our local module
using .MFS3D
include("utils.jl")
using LinearAlgebra
using Lebedev
using GLMakie
using Printf

@printf "test MFS3D grad u...\n"
x0 = [0.2,0.3,-0.5]'      # src point deep in unit sphere
for Na in 500:500:2000    # upper limits for N
    X, w = get_sphdesign(Na)
    N = length(w)                         # actual num sph pts
    ut,gradut = lap3dchgeval(X,x0,1.0)    # unit chg at x0
    flux = dot(w, sum(gradut.*X,dims=2))  # surf int u_n (nx=X)
    @printf "\tN=%d:\tflux err=%.3g\n" N flux-1.0
end

Na = 1202   # eg 1202 is common to both types
Ma = 3000
R = 0.7     # src pt radius
@printf "test sph MFS, compare proxy sph types (Na=%d)...\n" Na
sph_funcs = [get_sphdesign, get_lebedev]      # quad choices
sph_names = ["sph design", "Lebedev"]
fig = Figure(); ax = Axis(fig[1,1], xlabel=L"j", ylabel=L"\sigma_j", yscale=log10)
ax.title="Singular values of A matrix, 3D Laplace sphere, R=$R, two point choices"
for (i,sph_pts) in enumerate(sph_funcs)
    Y,_ = sph_pts(Na)
    Y *= R
    X, w = sph_pts(Ma)
    N,M = size(Y,1),length(w)
    @printf "\t%s with N=%d src and M=%d colloc\n" sph_names[i] N M
    A = lap3dchgpotmat(X,Y)
    sigs = svd(A).S
    scatterlines!(sigs, label=@sprintf "%s N=%d M=%d" sph_names[i] N M)
end
axislegend()
display(fig)

