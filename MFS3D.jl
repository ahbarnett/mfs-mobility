module MFS3D
export lap3dchgpotmat, lap3dchgeval

"""
A = lap3dchgpotmat(t, s)

Fill dense MFS matrix for 3D Laplace fundamental solution, plain
charges to potentials.
Kernel is 1/(4pi.r).

t target coords (M,3)
s source coords (N,3)
Output: A is target-from-source matrix (M,N)
"""
function lap3dchgpotmat(t, s)
    N = size(s,1)
    M = size(t,1)
	@assert size(s,2)==3  # check dim
	@assert size(t,2)==3
    A = similar(t, M, N)   # alloc output
    for i = 1:M
        for j = 1:N
            d1 = s[j,1] - t[i,1]
            d2 = s[j,2] - t[i,2]
            d3 = s[j,3] - t[i,3]
            rr = d1*d1 + d2*d2 + d3*d3
            A[i,j] = 1/(4pi*sqrt(rr))
        end
    end
	A
end

"""
u,gradu = lap3dchgeval(t, s, co)

Direct summation of 3D Laplace fundamental solutions, charges to potentials
and gradients. Kernel is 1/(4pi.r).

t target coords (M,3)
s source coords (N,3)
co coeff vector (N,)

Outputs: u is (M,), gradu is (M,3)
"""
function lap3dchgeval(t, s, co)
    N = size(s,1)
    M = size(t,1)     # num targs
	@assert size(s,2)==3  # check dim
	@assert size(t,2)==3
    u = zeros(M)   # assume Float64
	gradu = zeros(M,3)
    for i = 1:M
        for j = 1:N
            d1 = s[j,1] - t[i,1]
            d2 = s[j,2] - t[i,2] 
            d3 = s[j,3] - t[i,3]
            rr = d1*d1 + d2*d2 + d3*d3
            ir = 1.0/sqrt(rr)
            u[i] += co[j] * ir
			tmp = co[j] * ir*ir*ir
			gradu[i,1] += d1*tmp
			gradu[i,2] += d2*tmp
			gradu[i,3] += d3*tmp
        end
		u[i] *= 1/4pi
		gradu[i,:] *= 1/4pi
    end
	u, gradu
end

end  # module

