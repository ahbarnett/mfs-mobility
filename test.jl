# Play with MFS for 2D capacitance/elastance BVPs. Tests for now
# Barnett 11/3/23

include("MFS2D.jl")   # our local module
using .MFS2D
include("utils.jl")
using LinearAlgebra
using GLMakie
using Printf

verb = 1
N=60
M=2*N
t,tw,tn = unitcircle(M)   # colloc (targ) pts: log_cap=1 -> careful
s,_,_ = unitcircle(N)    # source pts
s *= 0.7

@printf "1-body exterior Dirichlet BVP with specified net charge Sig...\n"
A=lapchgmat(t,s)
A = [A; ones(1,N)]    # extra condition on net charge
A = [A [ones(M,); 0.0] ]   # append overall const as a dof (can't tie it to net chg)
println("N=$N, M=$M, last sing vals=",svd(A).S[end-2:end])
s0 = [1.1 0.8]        # incident source pt (row vec), not too close 
uinc = lapchgmat(t,s0)[:]   # unit chg,  [:] makes a col vec not M*1 matrix
Sig = 1.7          # desired net charge from this body (needn't match that of inc src)
rhs = [-uinc; Sig]
co = A\rhs
r=norm(A*co-rhs)/norm(rhs)
ut = lapchgeval(t,s,co[1:N]) .+ co[end]    # eval
re = norm(ut+uinc)/norm(uinc)              # rel surf error 
@printf "solved\trelresid=%.3g, bdryrelerr=%.3g, norm(c)=%.3g\n" r re norm(co)
@printf "\ttot charge err=%.3g\n" sum(co[1:N])-Sig
if verb>0       # plot soln and geom
    ng=300; g = range(-2,2,ng); o=ones(size(g)); gg=[kron(o,g) kron(g,o)]
    ugg = lapchgeval(gg,s,co[1:N]) .+ co[end]   # u_scatt on grid
    ugg += lapchgmat(gg,s0)[:]              # add in u_inc
    fig,ax,h = heatmap(g,g,reshape(ugg,ng,ng), axis=(aspect=DataAspect(),))
    h.colorrange=0.2*[-1,1]; h.colormap=:jet
    scatter!(s[:,1],s[:,2],color=:red)
    scatter!(t[:,1],t[:,2],color=:black)
    scatter!(s0[1],s0[2],color=:green)
    Colorbar(fig[1,2],h)
    fig   # don't need to display(fig) in vscode
end


