using DelimitedFiles         # for text file reading
using Printf
using Lebedev

"""
x,w,nx = unitcircle(N::Integer)
    
Periodic trapezoid rule quadrature for unit circle.
Returns N*2 node coords, N weights, N*2 unit normals.
The first node is at (1,0)
"""
function unitcircle(N::Integer)
    th = (0:N-1)/N*2*pi
    x = [cos.(th) sin.(th)]
    w = (2*pi/N)*ones(N)
    nx = copy(x)
    x,w,nx
end

"""
x,w = starfish(N::Integer=100,freq::Integer=5,ampl=0.3,rot=1.0)
    
Periodic trap rule quadrature for smooth starfish of a given frequency and
amplitude. Returns N*2 node coords, N weights. The first node is at theta=0.

*** should return a struct with xp, etc
"""
function starfish(N::Integer=100,freq::Integer=5,ampl=0.3,rot=1.0)
    th = (0:N-1)/N*2*pi                      # col vec
    r = 1.0 .+ ampl*cos.(freq*th .- rot)   # set of radii. Note .operators
    rp = -ampl*freq*sin.(freq*th .- rot)
    c = cos.(th); s = sin.(th)
    x = [r.*c r.*s]
    xp = [rp.*c.-r.*s rp.*s.+r.*c]  # velocity
    sp = sqrt.(sum(xp.^2,dims=2))        # speed at nodes
    w = (2*pi/N)*sp                   # PTR
    x,w
end

"""
    Ns = getavailablesphdesigns()

returns available list of numbers of spherical design points.
`Ns[t]` is the number of points in the spherical design exact for
degree `t`, for t=1,...,180.

Also see: [`get_sphdesign`](@ref)
"""
function getavailablesphdesigns()
    # file list with leading "sf" chars removed (dot is separator)...
    tN=readdlm(string(@__DIR__,"/sphdesigns/filelist.txt"),'.',Int)
    # (note linux, OSX, not Windows)
    tN[:,2]
end

"""
    X, w = get_sphdesign(Nmax)

Load then returns the largest spherical design (quadrature points on S^2) with
no more than `Nmax` points. `X` is the (N,3) coordinate array in R^3
of the N points, and `w`, a (N,) column vector of their corresponding
weights, w.r.t. surface measure on S^2. (The weights are equal, thus easily calculated by the user anyway.)
N<=Nmax, with N also <= 180, the largest design available in Wormseley's files.

Also see: [`getavailablesphdesigns`](@ref)
"""
function get_sphdesign(Nmax::Integer=1000)
    Ns = getavailablesphdesigns()
    t = findlast(N -> N<=Nmax, Ns)        # degree
    @assert !isnothing(t) "no available N are <= requested Nmax!"
    N = Ns[t]     # the largest N not exceeding Nmax
    fnam = @sprintf "sf%.3d.%.5d" t N     # reverse-engineer filename
    absfnam = string(@__DIR__,"/sphdesigns/",fnam)
    X = readdlm(absfnam)
    @assert size(X,1)==N "read wrong number of lines from file! Please see sphdesigns/README"
    w = ones(N)*(4pi/N)
    X,w
end

"""
    X, w = get_lebedev(Nmax)

Returns the largest Lebedev quadrature (points on S^2) with
no more than `Nmax` points, using the Lebedev.jl package.
`X` is the (N,3) coordinate array in R^3
of the N points, and `w`, a (N,) column vector of their corresponding
weights w.r.t. surface element on S^2. N<=Nmax.

Matches the behavior of [`get_sphdesign`](@ref)
"""
function get_lebedev(Nmax::Integer=1000)
    Ns = getavailablepoints()
    N = maximum(Ns[Ns .<= Nmax])
    @assert !isempty(N) "no available N are <= requested Nmax!"
    x,y,z,w = lebedev_by_points(N)
    X = [x y z]
    w = 4pi*w         # make w.r.t. surf measure
    X,w
end

"""
    X,w = get_fibonacci(N)

Returns `X` a (N,3) array of coordinates of `N` Fibonacci points on S^2
(the unit sphere), and `w` a (N,) vector of their equal weights w.r.t.
surface measure. `N` may be any positive integer.

Note: these weights are *not* high-order accurate for quadrature, since
such a set of weights is not discussed in literature for the Fibonacci point family.

See:
Richard Swinbank, James Purser, "Fibonacci grids: A novel approach to global modelling,"
Quarterly Journal of the Royal Meteorological Society, Volume 132, Number 619, July 2006 Part B, pages 1769-1793.

R. Marques, C. Bouville, K. Bouatouch, and J. Blat, "Extensible Spherical Fibonacci Grids," IEEE
Transactions on Visualization and Computer Graphics, vol. 27, no. 4, pp. 2341–2354, 2021 doi:
10.1109/TVCG.2019.2952131

Matches the behavior of [`get_sphdesign`](@ref)
"""
function get_fibonacci(N::Integer=1000)
    Phi = (1+sqrt(5.0))/2
    dphi = 2pi/Phi         # azimuth regular spacing
    X = zeros(N,3)
    for j=1:N              # loop over pts
        X[j,3] = 1 - (2j-1)/N       # z, symmetrically shifted
        phi = dphi*j                # offset irrelevant
        (s,c) = sincos(phi)
        rho = sqrt(1-X[j,3]^2)
        X[j,1] = rho*c
        X[j,2] = rho*s
    end
    w = (4pi/N)*ones(N)    # dummy weights for now
    X,w
end

mean(x) = sum(x)/length(x)       # not in base, weirdly

"""
    Xc = sphere_cluster_tree(K::Integer,delta; seed::Integer=0)

Returns `Xc` a (K,3) array of coordinates in R3 of centers of a cluster of `K` unit
spheres achieving minimum separation of `delta>0`. Dumb K^2 algorithm which attempts
to add sphere delta-near to a random sphere in a random direction, rejects if
intersects, until got enough.
"""
function sphere_cluster_tree(K::Integer,delta; seed::Integer=0)
    @assert delta>0
    Xc = zeros(K, 3)  # center coords of spheres
    k = 2             # index of next sphere to create
    Random.seed!(seed)
    while k <= K
        j = rand(1:k-1)     # pick random existing sphere
        v = randn(3)
        v *= (2 + delta) / norm(v)      # displacement vec
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
    return Xc
end

vecnorm(A::AbstractArray) = [norm(A[:,j],2) for j in eachcol(A)]  # a la Matlab

"""
    Xc = sphere_cluster_broms(K::Integer,delta; tol=1e-5, seed::Integer=0, maxit=100)

Returns `Xc` a (K,3) array of coordinates in R3 of centers of a cluster of `K` unit
spheres achieving minimum separation of `delta>0`. Anna Broms K^2 algorithm which
moves a sphere away from origin along a fixed random direction until is within
`tol/delta` of the right minumum distance from other spheres, until got enough.

Reimplements grow_cluster.m by Broms. 4/12/24, Barnett.
"""
function sphere_cluster_broms(K::Integer,delta; tol=1e-5, seed::Integer=0,
                              maxit::Integer=100)
    @assert delta>0
    Xc = zeros(K, 3)     # center coords of spheres
    k = 2                # leave 1st sphere at origin
    Random.seed!(seed)
    while k <= K
        v = randn(3)'     # row vec
        v /= norm(v)      # unit direction vec (Broms calls n)
        # Anna's function to rootfind on: desired dist to prior spheres
        f(s) = sqrt(minimum(sum((s*v .- Xc[1:k-1,:]).^2, dims=2))) - (2+delta)
        s2 = 2+delta; s1 = s2+0.1     # initial guess of distances along ray
        iter = 0
        f1 = f(s1); f2 = f(s2)
        while abs(f2)/delta > tol && iter<maxit
            snew = s2 - f2*(s2-s1)/(f2-f1)
            s1,s2 = s2,snew
            f1,f2 = f2,f(snew)               # one f eval per iter
            iter += 1
        end
        if iter<maxit
            Xc[k,:] = s2*v                   # keep that sphere
            k += 1
        else
            println("k=$k: reached maxit, trying new direction")
        end
    end
    return Xc
end
