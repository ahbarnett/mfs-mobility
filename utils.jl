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

Returns the largest spherical design (quadrature points on S^2) with
no more than `Nmax` points. `X` is the (N,3) coordinate array in R^3
of the N points, and `w`, a (N,) column vector of their corresponding
weights, w.r.t. surface measure on S^2. (The weights are equal, thus easily calculated by the user anyway.)
N<=Nmax, with N also <= 180, the largest design available.

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
    @assert size(X,1)==N "read wrong number of lines from file!"
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
