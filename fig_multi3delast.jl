# convergence plot for 3d spheres elastance/resistance.
# taken from multi3delast, without round-trip check.
# also collect # iters, overall time?  Barnett 2/20/24

include("MFS3D.jl")
using .MFS3D
include("utils.jl")
using LinearAlgebra       # dense stuff
using Random
using Krylov
using LinearMaps             # needed by Krylov
using CairoMakie
using Printf

verb = 1                     # verbosity (0=no figs, 1=figs, ...)
elast = true                 # false: solve Dir BVP. true: elastance BVP

Mratio = 1.2                 # approx M/N for MFS colloc/proxy
Rp = 0.7                    # proxy radius (eg 0.8 for d=1e-2, Na=4k)
K = 10                       # num spheres (keep small since (MK)^2 cost)
deltamin = 0.1               # min sphere separation; let's achieve it
sphvals = range(0.0,K-1)     # test data for v_k (Dir, fixed for C12 chk) or q_k (elast)
Nas = 200:200:2000                    # conv param (upper limit for N each time)
# (K=10, Nas=200:200:2000 takes ~2 min to run)

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

errs = collect(0.0*Nas)         # LNT style allocate
Ns = similar(Nas); iters = similar(Nas)
for (i,Na) in enumerate(Nas)    # ..............................................

# setup and solve 1 sphere...
Y,_ = get_sphdesign(Na)
Y *= Rp
N = size(Y, 1)                  # actual num proxy pts
Ns[i] = N                       # save it
X, w = get_sphdesign(Int(ceil(Mratio * N)))
M = length(w)
@printf "Na=%d.... (N=%d M=%d)........\n" Na N M
A = lap3dchgpotmat(X, Y)        # a.k.a. S (single-layer matrix)
if elast @printf "elastance case: perturbing S to S(I-L)+Lr ...\n"
    L = fill(1.0/N, (N,N))      # square 1s mat, I-L kills const
    Lr = fill(1.0/N, (M,N))     # rect 1s mat
    A = A*(I-L) + Lr            # perturb; unkn const V = -Lr.co
end
@time F = svd(A)
Z = F.Vt' * Diagonal(1 ./ F.S)   # so pseudoinv apply A^+ b = Z*(F.U'*b)
println("sphere: N=$N, M=$M, sing vals in ", extrema(F.S), " cond(A)=", F.S[1] / F.S[end])
b = X[:,2]              # a smooth test vec on surf
@printf "\tcheck pinv A works on smooth vec: %.3g\n" norm(A \ b - Z * (F.U' * b)) / norm(b)

# set up all KN proxy, all KM colloc, and (>KM) surf test nodes...
XX = zeros(K*M,3)    # all surf (colloc) nodes
YY = zeros(K*N,3)    # all source (proxy) nodes
Xt,_ = get_sphdesign(2*M); Mt = size(Xt,1); XXt = zeros(K*Mt,3)  # test nodes
for k=1:K            # copy in displaced sphere nodes
    XX[M*(k-1).+(1:M),:] = Xc[[k],:] .+ X    # note [k] to extract a row vec
    XXt[Mt*(k-1).+(1:Mt),:] = Xc[[k],:] .+ Xt
    YY[N*(k-1).+(1:N),:] = Xc[[k],:] .+ Y
end
if verb>1
    fig,ax,l = scatter(XX[:,1],XX[:,2],XX[:,3],color=1:K*M,markersize=3)
    zoom!(ax.scene,0.5); Label(fig[1,1,Top()], "colloc pts colored by index")
    display(fig)
end

@time AAoffdiag = lap3dchgpotmat(XX, YY)   # fill (!) full system mat Sij
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
    vs = sphvals
    # sign err was here: need rhs to be sphvals, not uinc (scatt prob)...
    uinc(x) = -vs[findmin(sum((Xc .- x').^2,dims=2))[2][1]]  # kth sph gets vs[k]
    rhs = -uinc.(eachrow(XX))          # eval uinc all surf nodes
    global matvec(g) = g + AAoffdiag*blkprecond(g)    # R-precond MFS sys
else        # elastance BVP. Use "completion" potential...
    chgs = sphvals
    co0 = kron(chgs,fill(1.0/N,N))       # stack completion dens
    u0(X) = lap3dchgeval(X,YY,co0)[1]    # completion pot evaluator
    rhs = -u0(XX)
    global matvec(g) = g + AAoffdiag*blkorthogL(blkprecond(g))  # R-precond MFS sys
end

matvecop = LinearMap(g -> matvec(g), M*K)  # hmm, has to be easier way :(
g,stats = gmres(matvecop, rhs; restart=false, rtol=1e-8, history=true, verbose=0)
@printf("GMRES done: niter=%d, relres=%.3g, gmax=%.3g, in %.3g sec\n", stats.niter,
    stats.residuals[end]/norm(rhs), norm(g,Inf), stats.timer)
    iters[i] = stats.niter

# check soln err by getting co, then direct eval @ test pts...
co = blkprecond(g)
if !elast          # Dir BVP
    chgs = [sum(co[N*(k-1).+(1:N)]) for k in 1:K]  # get q_k & save...
    uinct = uinc.(eachrow(XXt))      # u_inc @ test pts
    @time ut, gut = lap3dchgeval(XXt,YY,co)
    bcerr = uinct .+ ut              # the rep
    errs[i] = norm(bcerr,Inf)/norm(uinct,Inf)
    @printf "Dir rel max err at %d surf test pts: %.3g\n" K*Mt errs[i]
else               # elastance: eval the rep ut & chk voltages
    vs = [-mean(co[(k-1)*N.+(1:N)]) for k=1:K]    # voltages out 
    @time ut, gut = lap3dchgeval(XXt,YY,blkorthogL(co) .+ co0)  # the rep
    bcerr = ut .- kron(vs,ones(Mt))    # compare surf voltages
    errs[i] = norm(bcerr,Inf)/norm(vs,Inf)
    @printf "Elast rel max err at %d surf test pts: %.3g\n" K*Mt errs[i]
end
end                # .................................................

using DelimitedFiles
#writedlm("data/P10.d0.1.R0.7.dat", [Ns errs iters])
#data = readdlm("data/P10.d0.1.R0.7.dat")
#Ns = data[:,1]; errs = data[:,2]; iters=data[:,3]
# conv plot...
#Ns = [maximum(Naa[Naa.<Na]) for Na in Nas]        # if needed
fig=Figure(fontsize=20)
ax=Axis(fig[1,1],yscale=log10,xscale=sqrt,xlabel=L"N")
scatterlines!(Ns,errs,markersize=10,label=L"max resid err$$")
# image accum radius, from our resistance paper...
Racc = 1+deltamin/2 - sqrt(deltamin+deltamin^2/4)
lines!(Ns,0.2*Racc.^sqrt.(Ns),color=:green,linestyle=:dash,
    label=L"$O(R_\text{acc}^{\sqrt{N}})$")
axislegend()
display(fig)
save("pics/P10_d0.1_R0.7_resid_conv.pdf",fig)
