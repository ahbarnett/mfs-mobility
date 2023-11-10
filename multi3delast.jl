# test 3D multi-sphere (monodisperse) Dir/elastance BVPs
# via dense iterative solve of 1-body precond MFS. 
# Try L (1s-matrix, ie, orthog projector) and Lr perturbs.
# Incorporates multi3ddir.jl (via elast=false).
# Has roundtripchk: loads ans from other BVP & chks recovers known
# Barnett 11/09/23

include("MFS3D.jl")
using .MFS3D
include("utils.jl")
using LinearAlgebra       # dense stuff
using Krylov
using LinearMaps             # needed by Krylov
using GLMakie
using Printf
using Random                 # so can set seed

verb = 1                     # verbosity (0=no figs, 1=figs)
elast = true                 # false: solve Dir BVP. true: elastance BVP
roundtripchk = false         # false: use sphvals. true: load data (needs file)
Na = 1200                    # conv param (upper limit for N)
Mratio = 1.2                 # approx M/N for MFS colloc/proxy
Rp = 0.7                     # proxy radius
K = 10                       # num spheres (keep small since (MK)^2 cost)
deltamin = 0.1               # min sphere separation; let's achieve it
sphvals = range(1.0,K)       # test data for v_k (Dir) or q_k (elast)

# setup and solve 1 sphere...
Y,_ = get_sphdesign(Na)
Y *= Rp
N = size(Y, 1)                  # actual num proxy pts
X, w = get_sphdesign(Int(ceil(Mratio * N)))
M = length(w)
A = lap3dchgpotmat(X, Y)        # a.k.a. S (single-layer matrix)
if elast @printf "elastance case: perturbing S to S(I-L)+Lr ...\n"
    L = fill(1.0/N, (N,N))      # square 1s mat, I-L kills const
    Lr = fill(1.0/N, (M,N))     # rect 1s mat
    A = A*(I-L) + Lr            # perturb; unkn const V = -Lr.co
end
F = svd(A)
Z = F.Vt' * Diagonal(1 ./ F.S)   # so pseudoinv apply A^+ b = Z*(F.U'*b)
println("sphere: N=$N, M=$M, sing vals in ", extrema(F.S), " cond(A)=", F.S[1] / F.S[end])
b = X[:,2]              # a smooth test vec on surf
@printf "\tcheck pinv A works on smooth vec: %.3g\n" norm(A \ b - Z * (F.U' * b)) / norm(b)

# make cluster of K unit spheres, some pair separations=deltamin (dumb K^2 alg)
Xc = zeros(K, 3)  # center coords of spheres (todo: make func)
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
    zoom!(ax.scene,0.5); Label(fig[1,1,Top()], "colloc pts colored by index")
    display(fig)
end

AAoffdiag = lap3dchgpotmat(XX, YY)   # fill (!) full system mat Sij
# *** to do: make an applier without fill, or FMM wrapper.
for k=1:K      # kill each diag block
    AAoffdiag[M*(k-1).+(1:M),N*(k-1).+(1:N)] .= 0.0
end

"""
Apply block-diag preconditioner to length-MK surface vector `g` to
get length-NK MFS proxy coefficient vector `co`.
Uses globals `N`, `M`, `K`, `F.U` and `Z`, the last two for pinv(A).
Allocating output for now; who cares?
"""
function blkprecond(g)
    co = zeros(N*K)
    for k=1:K         # apply pinv(A) to each block
        co[N*(k-1).+(1:N)] = Z * (F.U' * g[M*(k-1).+(1:M)])
    end
    co
end

function blkorthogL(co)  # apply (I-L) to each block, allocating
    out = similar(co)
    for k=1:K
        blk = N*(k-1).+(1:N)         # index set of block
        out[blk] = co[blk] .- mean(co[blk])  # equiv to (I-L)co
    end
    out
end

if !elast   # Dirichlet BVP; pick data *** to make backgnd uinc switchable?
    #uinc(x) = x[3]  # incident (applied) pot, expects 3-vec. Efield=(0,0,-1)
    if roundtripchk vs=readdlm("data/multi3d_pots.dat")    # output of elastance
    else vs = sphvals end
    # sign err was here: need rhs to be sphvals, not uinc (scatt prob)...
    uinc(x) = -vs[findmin(sum((Xc .- x').^2,dims=2))[2][1]]  # kth sph gets vs[k]
    rhs = -uinc.(eachrow(XX))          # eval uinc all surf nodes
    matvec(g) = g + AAoffdiag*blkprecond(g)    # R-precond MFS sys
else        # elastance BVP. Use "completion" potential...
    if roundtripchk chgs=readdlm("data/multi3d_chgs.dat")   # output of Dir BVP
    else chgs = sphvals end
    co0 = kron(chgs,fill(1.0/N,N))       # stack completion dens
    u0(X) = lap3dchgeval(X,YY,co0)[1]    # completion pot evaluator
    rhs = -u0(XX)
    matvec(g) = g + AAoffdiag*blkorthogL(blkprecond(g))  # R-precond MFS sys
end
matvecop = LinearMap(g -> matvec(g), M*K)  # hmm, has to be easier way :(
g,stats = gmres(matvecop, rhs; restart=false, rtol=1e-6, history=true, verbose=0)
@printf "GMRES done: niter=%d, relres=%.3g, in %.3g sec\n" stats.niter stats.residuals[end]/norm(rhs) stats.timer
# check soln err by getting co, then direct eval @ test pts...
co = blkprecond(g)
if !elast          # Dir BVP
    chgs = [sum(co[N*(k-1).+(1:N)]) for k in 1:K]  # get q_k & save...
    if roundtripchk        # compare against known ans from other BVP type...
        @printf "roundtripchk: max rel q_k err %.3g\n" norm(sphvals.-chgs,Inf)/norm(sphvals,Inf)
    else writedlm("data/multi3d_chgs.dat", chgs)   # in pkg via utils
    end
    uinct = uinc.(eachrow(XXt))      # u_inc @ test pts
    @time bcerr = uinct .+ lap3dchgeval(XXt,YY,co)[1]  # the rep
    @printf "Dir rel max err at %d surf test pts: %.3g\n" K*Mt norm(bcerr,Inf)/norm(uinct,Inf)
else               # elastance: eval the rep ut & chk voltages
    vs = [-mean(co[(k-1)*N.+(1:N)]) for k=1:K]    # voltages out 
    if roundtripchk
        @printf "roundtripchk: max rel v_k err %.3g\n" norm(sphvals.-vs,Inf)/norm(sphvals,Inf)

    else writedlm("data/multi3d_pots.dat", vs)
    end
    @time ut = lap3dchgeval(XXt,YY,blkorthogL(co) .+ co0)[1]  # the rep
    bcerr = ut .- kron(vs,ones(Mt))    # compare surf voltages
    @printf "Elast rel max err at %d surf test pts: %.3g\n" K*Mt norm(bcerr,Inf)/norm(vs,Inf)
end
if verb>0
    GLMakie.activate!(title="multi3d: BC residuals @ test pts")
    fig2,ax2,l2 = scatter(XXt[:,1],XXt[:,2],XXt[:,3],color=bcerr,markersize=5)
    l2.colorrange = norm(bcerr,Inf)*[-1,1]     # symmetric colors
    l2.colormap=:jet; Colorbar(fig2[1,2],l2)
    zoom!(ax2.scene,0.5)
    display(GLMakie.Screen(), fig2)
end
