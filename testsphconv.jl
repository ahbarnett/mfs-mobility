# convergence test for sphere points, two types. Barnett 11/06/23

using LinearAlgebra
using Lebedev
include("utils.jl")

kvec = 10 * [6.0,3.0,2.0]    # wavevec in R^3, Pythagorean quadruple.
k = norm(kvec)               # note length of [2,3,6] is 7.
f(x::AbstractVector) = cos(dot(kvec, x))    # func on S^2 to integrate
Ie = 4pi * sin(k) / k   # exact integral (by rotating so z parallel to a)

quaderr(X, w) = dot([f(x) for x in eachrow(X)], w) - Ie

Nmax = 6000
LNs = getavailablepoints()         # test Lebedev pts...
LNs = LNs[LNs .<= Nmax]
Lerrs = zeros(length(LNs))
for (t, N) in enumerate(LNs)
    x, y, z, w = lebedev_by_points(N)
    w *= 4pi                       # Lebedev weights for avg not area
    Lerrs[t] = quaderr([x y z],w)
end
fig,ax,l = scatterlines(LNs,abs.(Lerrs),label="Lebedev")
ax.limits=(0,Nmax,1e-15,1e1)
ax.yscale=log10; ax.xlabel=L"N"; ax.ylabel="quadr error"

DNs = getavailablesphdesigns()     # test spherical designs...
DNs = DNs[DNs .<= Nmax]
Derrs = zeros(length(DNs))
for (t, N) in enumerate(DNs)
    X, w = sphdesign_by_points(N)
    Derrs[t] = quaderr(X,w)
end
scatterlines!(DNs,abs.(Derrs),label="Sph designs")
lines!(k^2/3*ones(2),[1.0, 1e-15],label="k^2/3",color=:red)
axislegend()
ax.title=@sprintf "Quadrature convergence comparison for oscillatory func on S^2: k=%.3g" k
display(fig)

