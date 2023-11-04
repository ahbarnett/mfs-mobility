module MFS2D
export lapchgpotmat, lapchgeval
"""
A = lapchgpotmat(t, s)

Fill dense MFS matrix for 2D Laplace fundamental solution, plain
charges to potentials.
Kernel is (1/2pi) log(1/r). No augmentation by any const term.

t is M*2 target coords, s is N*2 source coords.
A is M*N
"""
function lapchgpotmat(t, s)
    N = size(s,1)
    M = size(t,1)
	@assert size(s,2)==2  # check dim
	@assert size(t,2)==2
    A = similar(t, M, N)   # alloc output
	# dumb loop; consider eachrow(t) in a [... for ...] construct
    for i = 1:M
        for j = 1:N
            d1 = s[j,1] - t[i,1]
            d2 = s[j,2] - t[i,2]
            rr = d1*d1 + d2*d2    # squared dist
            A[i,j] = (-1/4pi) * log(rr)
        end
    end
	A
end

"""
u,gradu = lapchgeval(t, s, co)

Direct summation of 2D Laplace fundamental solutions, charges to potentials
and gradients. Kernel is (1/2pi) log(1/r). No const term.

t is M*2 target coords, s is N*2 source coords, co is (N,) coeff vec
Outputs: u is (M,), gradu is (M,2)
"""
function lapchgeval(t, s, co)
    N = size(s,1)
    M = size(t,1)     # num targs
	@assert size(s,2)==2  # check dim
	@assert size(t,2)==2
    u = zeros(M)   # assume Float64
	gradu = zeros(M,2)
    for i = 1:M
        for j = 1:N
            d1 = s[j,1] - t[i,1]
            d2 = s[j,2] - t[i,2]
            rr = d1*d1 + d2*d2
            u[i] += co[j]*log(rr)
			tmp = co[j]/rr
			gradu[i,1] += d1*tmp
			gradu[i,2] += d2*tmp
        end
		u[i] *= -1/4pi
		gradu[i,:] *= -1/2pi
    end
	u, gradu
end

end  # module
