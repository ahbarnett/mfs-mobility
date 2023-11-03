module MFS2D
export lapchgmat, lapchgeval
"""
A = lapchgmat(t, s)

Fill dense MFS matrix for 2D Laplace fundamental solution, plain
charges to potentials.
Kernel is (1/2pi) log(1/r). No augmentation by any const term.

t is M*2 target coords, s is N*2 source coords.
A is M*N
"""
function lapchgmat(t, s)
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
            r2 = d1*d1 + d2*d2    # squared dist
            A[i,j] = (-1/4pi) * log(r2)
        end
    end
	A
end

"""
u = lapchgeval(t, s, co)

Direct summation of 2D Laplace fundamental solutions, charges to potentials
Kernel is (1/2pi) log(1/r). No const term.

t is M*2 target coords, s is N*2 source coords, co is (N,) coeff vec
Output u is (M,)
"""
function lapchgeval(t, s, co)
    N = size(s,1)
    M = size(t,1)     # num targs
	@assert size(s,2)==2  # check dim
	@assert size(t,2)==2
    u = zeros(M)   # assume Float64
    for i = 1:M
        for j = 1:N
            d1 = s[j,1] - t[i,1]
            d2 = s[j,2] - t[i,2]
            r2 = d1*d1 + d2*d2
            u[i] += co[j] * log(r2)
        end
		u[i] *= -1/4pi
    end
	u
end

end  # module
