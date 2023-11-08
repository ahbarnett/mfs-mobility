# 3D unit sphere Laplace ext Dir BVP via MFS. Barnett 11/7/23

include("MFS3D.jl")   # our local module
using .MFS3D
include("utils.jl")
using LinearAlgebra
using GLMakie
using Printf

x0 = [1.1,0.8,-0.4]'      # u inc src point far from unit sphere
R = 0.7                   # MFS source radius
@printf "1-body ext Lap Dir 3D conv (scatt BVP, src dist %.3g, R=%g)...\n" norm(x0) R
sph_pts = get_sphdesign   # which type of surf discr
Nas = 500:500:2000        # conv study: upper limits for N
Mratio = 1.2              # desired M/N
Ns = similar(Nas)
for (i,Na) in enumerate(Nas)
    Y,_ = sph_pts(Na)
    Y *= R
    N = Ns[i] = size(Y,1)                 # actual num sph pts
    X, w = sph_pts(Int(ceil(Mratio*N)))
    M = length(w)
    A = lap3dchgpotmat(X,Y)
    uinc = lap3dchgpotmat(X,x0)[:]   # unit chg,  [:] makes col vec not mat
    rhs = -uinc                           # Dirichlet BVP data
    co = A\rhs
    r=norm(A*co-rhs)/norm(rhs)
    ut,gradut = lap3dchgeval(X,Y,co)      # MFS eval on surf
    re = norm(ut+uinc)/norm(uinc)         # rel surf pot error 
    @printf "N=%d M=%d relresid=%.3g bdryrelerr=%.3g norm(c)=%.3g\n" N M r re norm(co)
    flux = dot(w,sum(gradut.*X,dims=2))   # surf int u_n (uinc_n flux=0)
    @printf "\ttot chg=%.8g, u_n flux=%.8g\n" sum(co) flux
end
