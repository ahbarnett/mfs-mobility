# test MFS3D

include("MFS3D.jl")   # our local module
using .MFS3D
include("utils.jl")
using LinearAlgebra
using GLMakie
using Printf

@printf "test MFS3D grad u...\n"
x0 = [0.2,0.3,-0.5]'      # src point in unit sphere
for Na in 500:500:2000
    X, w = sphdesign_by_points(Na)
    N = length(w)                        # actual num sph pts
    ut,gradut = lap3dchgeval(X,x0,1.0)    # unit chg at x0
    flux = dot(w, sum(gradut.*X,dims=2))   # surf int u_n
    @printf "\tN=%d:\tflux err=%.3g\n" N flux-1.0
end
