# examine 3D sphere points
# Barnett 11/06/23

using GLMakie
fig = Figure()

using Lebedev
# not quasi-uniform, since variable weights
# but Gaussian efficiency wrt degree
Ns = getavailablepoints()
N = Ns[20]   # 1202 pts
x,y,z,w = lebedev_by_points(N)
X = [x y z]
println("max to min weight ratio: ",maximum(w)/minimum(w))
# large dyn range (~8), small at octahedron corners
ax,l = scatter(fig[1,1], X[:,1],X[:,2],X[:,3],color=w)
fac = 0.4; zoom!(ax.scene,fac)
Label(fig[1,1,Top()], "Lebedev quadrature pts")   # only way to title 3d :(
l.colormap=:jet; Colorbar(fig[1,2],l)
sum(w)   # 1
w .*= 4pi   # make weights wrt surf element
println(sum(w))

# try spherical designs (2/3 Gaussian efficiency wrt degree, but quasi-unif)
using DelimitedFiles
Xd = readdlm("sphdesigns/sf048.01202")  # same N as above
Nd = size(Xd,1)
wd = ones(Nd)*4pi/Nd    # make own weights
println(sum(wd))        # exact
ax2,l2 = scatter(fig[2,1], Xd[:,1],Xd[:,2],Xd[:,3],color=:black,markersize=5)
zoom!(ax2.scene,fac)
Label(fig[2,1,Top()], "Sph design pts")   # only way to title 3d :(

# Fibonacci pts, not a good quadrature
include("utils.jl")
Xf,_ = get_fibonacci(1202)
ax3,l3 = scatter(fig[2,3], Xf[:,1],Xf[:,2],Xf[:,3],color=:black,markersize=5)
zoom!(ax3.scene,fac)
Label(fig[2,3,Top()], "Fibonacci pts")   # only way to title 3d :(
l.color=w   # recover colors in [1,1] ... a bug
display(fig)
#GLMakie.closeall()

