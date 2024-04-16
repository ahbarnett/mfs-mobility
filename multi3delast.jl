# test 3D multi-sphere (monodisperse) Dir/elastance BVPs
# via dense iterative solve of 1-body precond MFS. 
# Try L (1s-matrix, ie, orthog projector) and Lr perturbs.
# Incorporates multi3ddir.jl (via elast=false).
# Has roundtripchk: loads ans from other BVP & chks recovers known
# Barnett 11/09/23. Use for journal fig plots 2/20/23

include("MFS3D.jl")
using .MFS3D
include("utils.jl")
using LinearAlgebra       # dense stuff
using Krylov
using LinearMaps             # needed by Krylov
using GLMakie                # CairoMakie fails to z-buffer 3d scatter correctly
include("parula.jl")
using Printf
using Random                 # so can set seed

verb = 2                     # verbosity (0=no figs, 1=figs, ...)
elast = true                 # false: solve Dir BVP. true: elastance BVP
roundtripchk = false         # false: use sphvals. true: load data (needs file)
Na = 1000                    # conv param (upper limit for N)
Mratio = 1.2                 # approx M/N for MFS colloc/proxy
Rp = 0.7                    # proxy radius (eg 0.8 for d=1e-2, Na=4k)
K = 10                       # num spheres (keep small since (MK)^2 cost)
deltamin = 0.1               # min sphere separation; let's achieve it
sphvals = range(0.0,K-1)     # test data for v_k (Dir, fixed for C12 chk) or q_k (elast)

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
@time F = svd(A)
Z = F.Vt' * Diagonal(1 ./ F.S)   # so pseudoinv apply A^+ b = Z*(F.U'*b)
println("sphere: N=$N, M=$M, sing vals in ", extrema(F.S), " cond(A)=", F.S[1] / F.S[end])
b = X[:,2]              # a smooth test vec on surf
@printf "\tcheck pinv A works on smooth vec: %.3g\n" norm(A \ b - Z * (F.U' * b)) / norm(b)

# make cluster of K unit spheres, some pair separations=deltamin (dumb K^2 alg)
Xc = sphere_cluster_broms(K,deltamin)

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
g,stats = gmres(matvecop, rhs; restart=false, rtol=1e-7, history=true, verbose=0)
@printf("GMRES done: niter=%d, relres=%.3g, gmax=%.3g, in %.3g sec\n", stats.niter,
    stats.residuals[end]/norm(rhs), norm(g,Inf), stats.timer)
# check soln err by getting co, then direct eval @ test pts...
co = blkprecond(g)
if !elast          # Dir BVP
    chgs = [sum(co[N*(k-1).+(1:N)]) for k in 1:K]  # get q_k & save...
    if roundtripchk        # compare against known ans from other BVP type...
        @printf "roundtripchk: max rel q_k err %.3g\n" norm(sphvals.-chgs,Inf)/norm(sphvals,Inf)
    else writedlm("data/multi3d_chgs.dat", chgs)   # in pkg via utils
    end
    uinct = uinc.(eachrow(XXt))      # u_inc @ test pts
    @time ut, gut = lap3dchgeval(XXt,YY,co)
    bcerr = uinct .+ ut              # the rep
    @printf "Dir rel max err at %d surf test pts: %.3g\n" K*Mt norm(bcerr,Inf)/norm(uinct,Inf)
else               # elastance: eval the rep ut & chk voltages
    vs = [-mean(co[(k-1)*N.+(1:N)]) for k=1:K]    # voltages out 
    if roundtripchk
        @printf "roundtripchk: max rel v_k err %.3g\n" norm(sphvals.-vs,Inf)/norm(sphvals,Inf)

    else writedlm("data/multi3d_pots.dat", vs)
    end
    @time ut, gut = lap3dchgeval(XXt,YY,blkorthogL(co) .+ co0)  # the rep
    bcerr = ut .- kron(vs,ones(Mt))    # compare surf voltages
    @printf "Elast rel max err at %d surf test pts: %.3g\n" K*Mt norm(bcerr,Inf)/norm(vs,Inf)
end
if verb>0
    GLMakie.activate!(title="multi3d: BC residuals @ test pts")
    fig2 = Figure(fontsize=20,size=(700,500))
    a2 = LScene(fig2[1,1])
#    fig2,a2,l2 = scatter(XXt[:,1],XXt[:,2],XXt[:,3],color=bcerr,markersize=5)
#    l2.colorrange = norm(bcerr,Inf)*[-1,1]     # symmetric colors, linear colorscale
#    l2.colormap=:jet;
    l2 = scatter!(a2,XXt[:,1],XXt[:,2],XXt[:,3],color=abs.(bcerr),colorscale=log10,
        markersize=7,fxaa=true)
    # issue of white outline (overdraw=true ruins z-buffer)...
    #https://discourse.julialang.org/t/hide-stroke-in-3d-scatter-using-makie-jl/106383
    l2.colorrange = norm(bcerr,Inf)*[1e-3,1]     # log (top 3 digits of resid err)
    l2.colormap=parula  #  from parula.jl    #Reverse(:viridis);  # cute
    Colorbar(fig2[1,2],l2)
    #cam = cameracontrols(lsc2.scene)
    #cam.eyeposition[] = [2.0,0,0]; cam.fov[]=20    # doesn't change stuff!
    display(GLMakie.Screen(), fig2)
    zoom!(a2.scene,0.5)    # has no effect :(   ... is undone at the save stage!
    #update_cam!(a2.scene, cameracontrols(a2.scene))
    # had to view from below to get scatter point 3d z-buffer to look ok :(
    update_cam!(a2.scene, 0, -0.4)    # view angle (phi, theta) in radius
    save("pics/resid_P10_d0.1_N1000b.png",fig2; update=false)    # <- crucial!

    GLMakie.activate!(title="multi3d: u_n (charge dens) @ test pts")
    fig3 = Figure(fontsize=20,size=(700,500))
    a3 = LScene(fig3[1,1])
    unt = sum(gut.*kron(ones(K),Xt), dims=2)    # grad u dot n @ test pts
    l3 = scatter!(a3,XXt[:,1],XXt[:,2],XXt[:,3],color=unt[:],
        markersize=7,fxaa=true)
    l3.colorrange = norm(unt,Inf)*[-1,1]      # symmetric colors
    l3.colormap=:jet; Colorbar(fig3[1,2],l3)
    display(GLMakie.Screen(), fig3)
    zoom!(a3.scene,0.5)
    update_cam!(a3.scene, 0, -0.4)
    save("pics/un_P10_d0.1_N1000b.png",fig3; update=false)       # <- crucial!
end

if K==2 && !elast && !roundtripchk  # chk analytic capacitance (Lebedev et al '65 as in Cheng'01)
    beta = acosh(1+deltamin/2)    # acosh(l), since we built delta=deltamin
    # number of terms n here grows like delta^{-1/2}. 100 enough for 1e-2:
    C12 = sinh(beta) * sum([exp(-(2n+1)beta)/sinh((2n+1)beta) for n=0:100])
    @printf "C12_anal=%.8g: our q_1 rel err %.8g\n" C12 C12+chgs[1]/4pi
    # note C12 is *not* chgs[1] when antisymm voltage vs=[-1 1]. Need [0 1].
end

