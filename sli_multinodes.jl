# pic of MFS sources and colloc nodes for slides. Barnett 4/14/25
include("MFS3D.jl")
using .MFS3D
include("utils.jl")
using LinearAlgebra       # dense stuff
using Random
using CairoMakie
K=20
Rp=0.7
deltamin = 0.1
# make cluster of K unit spheres, some pair separations=deltamin (dumb K^2 alg)
Xc = sphere_cluster_broms(K,deltamin)
Na=400
Y,_ = get_sphdesign(Na)
Y *= Rp
N = size(Y, 1)                  # actual num proxy pts
X, w = get_sphdesign(Int(ceil(Mratio * N)))
M = length(w)
@printf "Na=%d.... (N=%d M=%d)........\n" Na N M
# set up all KN proxy, all KM colloc, and (>KM) surf test nodes...
XX = zeros(K*M,3)    # all surf (colloc) nodes
YY = zeros(K*N,3)    # all source (proxy) nodes
for k=1:K            # copy in displaced sphere nodes
    XX[M*(k-1).+(1:M),:] = Xc[[k],:] .+ X    # note [k] to extract a row vec
    YY[N*(k-1).+(1:N),:] = Xc[[k],:] .+ Y
end
#fig = Figure(size=(500,500))
#ax = Axis3(fig[1,1]; aspect=(1,1,1))    # aspect fails: spheres skew!
#scatter!(XX[:,1],XX[:,2],XX[:,3],color=:black,markersize=1)
fig, ax, l = scatter(figure=(size=(500,500),), XX[:,1],XX[:,2],XX[:,3],color=:black,markersize=1)
scatter!(YY[:,1],YY[:,2],YY[:,3],color=:blue,markersize=2)
#zoom!(ax.scene,0.5);
ax.show_axis=false
#hidedecorations!(ax)
#Label(fig[1,1,Top()], "colloc pts colored by index")
display(fig)
save("pics/sli_P20_multinodes.png",fig)

