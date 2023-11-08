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

Also see: sphdesign_by_points
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
        X[j,3] = 1 - (2j-2)/N       # z, not symmetrically shifted
        phi = dphi*(j-1)            # matching Broms
        (s,c) = sincos(phi)
        rho = sqrt(1-X[j,3]^2)
        X[j,1] = rho*c
        X[j,2] = rho*s
    end
    w = (4pi/N)*ones(N)    # dummy weights for now
    X,w
end


