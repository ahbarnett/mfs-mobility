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
