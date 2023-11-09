# test 3D multi-sphere (monodisperse) Dir BVP via 1-body MFS precond
# Barnett 11/08/23

include("MFS3D.jl")
using .MFS3D
include("utils.jl")
using LinearAlgebra       # dense stuff
using Krylov
using LinearMaps             # needed by Krylov
using GLMakie
using Printf
using Random                  # so can set seed

verb = 1  # verbosity
# setup and solve 1 sphere
Na = 1000       # upper limit for N, conv param
Mratio = 1.2
Rp = 0.7
reps = 1e-14    # pseudoinv regularization cutoff
Y, _ = get_sphdesign(Na)
Y *= Rp
N = size(Y, 1)                  # actual num proxy pts
X, w = get_sphdesign(Int(ceil(Mratio * N)))
M = length(w)
A = lap3dchgpotmat(X, Y)
F = svd(A)
#rankA = sum(F.S>reps)   # *** to do
Z = F.Vt' * Diagonal(1 ./ F.S)    # so pseudoinv applies via A^+ b = Z*(F.U'*b)
println("sphere: N=$N, M=$M, sing vals rng ", extrema(F.S), " cond(A)=", F.S[1] / F.S[end])
b = X[:,2]              # a smooth test vec on surf
@printf "\tcheck pinv A works on smooth vec: %.3g\n" norm(A \ b - Z * (F.U' * b)) / norm(b)

K = 5   # make cluster of K unit spheres near each other (dumb K^2 alg)
deltamin = 0.2    # min sphere separation; let's achieve it
Xc = zeros(K, 3)  # center coords of spheres
k = 2             # index of next sphere to create
Random.seed!(0)
while k <= K
    j = rand(1:k-1)     # pick random existing sphere
    v = randn(3)
    v *= (2 + deltamin) / norm(v)   # displacement vec
    trialc = (Xc[j, :] + v)'        # new center, row vec
    mindist = Inf
    if k > 2                     # exist others to check dists...
        otherXc = Xc[(1:K.!=j) .& (1:K.<k), :]   # exclude sphere j
        #println("k=$k, j=$j, o=",otherXc, ", tc=", trialc)
        mindist = sqrt(minimum(sum((trialc .- otherXc).^2, dims=2)))
    end
    if mindist >= 2 + deltamin
        Xc[k,:] = trialc            # keep that sphere
        k += 1
    end                             # else try again...
end

# set up all proxy, all colloc, and surf test nodes...
XX = zeros(K*M,3)    # all surf (colloc) nodes
YY = zeros(K*N,3)    # all source (proxy) nodes
Xt,_ = get_sphdesign(2*M); Mt = size(Xt,1); XXt = zeros(K*Mt,3)  # test nodes
for k=1:K            # copy in displaced sphere nodes
    XX[M*(k-1).+(1:M),:] = Xc[[k],:] .+ X    # note [k] to extract a row vec
    XXt[Mt*(k-1).+(1:Mt),:] = Xc[[k],:] .+ Xt
    YY[N*(k-1).+(1:N),:] = Xc[[k],:] .+ Y
end
if verb>0
    fig,ax,l = scatter(XX[:,1],XX[:,2],XX[:,3],color=1:K*M,markersize=3)
    zoom!(ax.scene,0.5)
end

AAoffdiag = lap3dchgpotmat(XX, YY)    # full dense system mat
# *** to do: make an applier without fill, or FMM wrapper.
for k=1:K      # kill each diag block
    AAoffdiag[M*(k-1).+(1:M),N*(k-1).+(1:N)] .= 0
end

"""
Apply block-diag preconditioner to length-MK surface vector `g` to
get length-NK MFS proxy coefficient vector `co`.
Uses global vars `N`, `M`, `K`, `F.U` and `Z`, the last two for pinv(A).
Allocating output for now.
"""
function blkprecond(g)
    co = zeros(N*K)
    for k=1:K         # apply pinv(A) to each block
        co[N*(k-1).+(1:N)] = Z * (F.U' * g[M*(k-1).+(1:M)])
    end
    co
end

uinc(x) = x[3]  # incident (applied) pot, expects 3-vec. Efield=(0,0,-1)
rhs = -uinc.(eachrow(XX))          # eval uinc all surf nodes
matvec(g) = g + AAoffdiag*blkprecond(g)    # R-precond MFS trick
matvecop = LinearMap(g -> matvec(g), M*K)  # hmm, has to be easier way :(
g,stats = gmres(matvecop, rhs; restart=false, rtol=1e-6, history=true, verbose=0)
@printf "GMRES done: niter=%d, relres=%.3g, in %.3g sec\n" stats.niter stats.residuals[end]/norm(rhs) stats.timer
# check soln err by getting co then using for direct eval...
co = blkprecond(g)
uinct = uinc.(eachrow(XXt))
bcerr = uinct .+ lap3dchgeval(XXt,YY,co)[1]
@printf "rel max err at %d surf test pts: %.3g\n" K*Mt norm(bcerr,Inf)/norm(uinct,Inf)
