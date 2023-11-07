# examine 3D sphere points
# Barnett 11/06/23

using GLMakie

using Lebedev
# not quasi-uniform, since variable weights
# but Gaussian efficiency wrt degree
Ns = getavailablepoints()
N = Ns[20]   # 1202 pts
x,y,z,w = lebedev_by_points(N)
X = [x y z]
println("max to min weight ratio: ",maximum(w)/minimum(w))
# large dyn range (~8), small at octahedron corners
GLMakie.activate!(title="Lebedev pts")
fig,ax,l = scatter(X[:,1],X[:,2],X[:,3],color=w)
l.colormap=:jet; Colorbar(fig[1,2],l)
display(fig)            # to see fig in vscode, since not last
sum(w)   # 1
w .*= 4pi   # make weights wrt surf element
println(sum(w))

# try spherical designs (2/3 Gaussian efficiency wrt degree, but quasi-unif)
using DelimitedFiles
Xd = readdlm("sphdesigns/sf048.01202")  # same N as above
Nd = size(Xd,1)
wd = ones(Nd)*4pi/Nd    # make own weights
println(sum(wd))        # exact
GLMakie.activate!(title="Wormseley sph design pts")
fig,ax,l = scatter(Xd[:,1],Xd[:,2],Xd[:,3],color=:black,size=1)
display(GLMakie.Screen(), fig)           # new figure
#GLMakie.closeall()

