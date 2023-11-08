# test 3D multi-sphere (monodisperse) Dir BVP via 1-body MFS precond
# Barnett 11/08/23

include("MFS3D.jl")
using .MFS3D
include("utils.jl")
using LinearAlgebra
using IterativeSolvers
using GLMakie
using Printf
using Random

# setup and solve 1 sphere
Na = 1000       # upper limit for N, conv param
Mratio = 1.2
Rp = 0.7
reps = 1e-14    # pseudoinv regularization cutoff
Y, _ = get_sphdesign(Na)
Y *= Rp
N = size(Y, 1)                  # actual num sph pts
X, w = get_sphdesign(Int(ceil(Mratio * N)))
M = length(w)
A = lap3dchgpotmat(X, Y)
F = svd(A)
#rankA = sum(F.S>reps)   # *** to do
Z = F.Vt' * Diagonal(1 ./ F.S)    # so pseudoinv applies via A^+ b = Z*(F.U'*b)
println("sphere: N=$N, M=$M, sing vals rng ", extrema(F.S), " cond(A)=", F.S[1] / F.S[end])
b = randn(M)
@printf "\tcheck pinv A works on rand vec: %.3g\n" norm(A \ b - Z * (F.U' * b)) / norm(b)

K = 3   # set up K unit spheres near each other (dumb K^2 alg)
deltamin = 0.2    # min sphere separation; let's achieve it
Xc = zeros(K, 3)  # center coords of spheres
k = 2             # index of next sphere to create
Random.seed!(0)
while k <= K
    j = rand(1:k-1)     # pick random existing sphere
    v = randn(3)
    v *= (2 + deltamin) / norm(v)   # displacement vec
    trialc = Xc[j, :] + v           # attempt new center
    show(trialc)
    mindist = Inf
    if k > 2                     # otherwise no others
        otherXc = Xc[1:(k-1).!=j, :]     # exclude sphere j
        mindist = sqrt(minimum(sum((trialc .- otherXc).^2, dims=2)))
    end
    if mindist >= 2 + deltamin
        Xc = [Xc; trialc]
        k += 1
    end
end
println(Xc)




