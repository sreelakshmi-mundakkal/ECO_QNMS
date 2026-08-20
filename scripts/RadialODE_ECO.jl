 """
Ordinary differential equation in radial direction for the
hyperboloidally compactified Teukolsky equation.
"""
module RadialODE_ECO

export eig_vals_vecs, Basis

include("Cheb.jl")
import .Cheb as CH

using LinearAlgebra
using GenericSchur
using SparseArrays

using ApproxFun


function radial_discretized_eqn_a_bordered(
    nr::Integer,
    s::Integer,
    m::Integer,
    a::T,
    bhm::T,
    om::Complex{T},
    rmin::T,
    rmax::T,
) where {T<:Real}
    d = rmin .. rmax
    D1 = Derivative(d, 1)
    D2 = D1^2
    x = Fun(identity, d)

    A = (
        (2 * im) * om - 2 * (1 + s) * x +
        2 * (im * om * ((a^2) - 8 * (bhm^2)) + im * m * a + (s + 3) * bhm) * x^2 +
        4 * (2 * im * om * bhm - 1) * (a^2) * x^3
    )
    B = (
        (((a^2) - 16 * (bhm^2)) * (om^2) + 2 * (m * a + 2 * im * s * bhm) * om) +
        2 *
        (
            4 * ((a^2) - 4 * (bhm^2)) * bhm * (om^2) +
            (4 * m * a * bhm - 4 * im * (s + 2) * (bhm^2) + im * (a^2)) * om +
            im * m * a +
            (s + 1) * bhm
        ) *
        x +
        2 * (8 * (bhm^2) * (om^2) + 6 * im * bhm * om - 1) * (a^2) * x^2
    )
    L = -(x^2 - 2 * bhm * x^3 + (a^2) * x^4)*D2+A*D1+B
    Bc = Evaluation(Chebyshev(d), rmax)

    
    Afull = Matrix(([Bc; L])[1:nr, 1:nr])

    preL = inv(Matrix(Conversion(Chebyshev(d), Ultraspherical(2, d))[1:nr-1, 1:nr-1]))

    
    LHS = vcat(Afull[1:1, :], preL * Afull[2:end, :])

    

    RHS = Matrix{Complex{T}}(I, nr, nr)
    RHS[1, :] .= 0 
    return LHS,RHS 
end


"""
    eig_vals_vecs(
            nr::Integer,
            s::Integer,
            m::Integer,
            a::T,
            om::Complex{T}
        ) where T<:Real

Compute eigenvectors and eigenvalues for the radial equation
using a pseudospectral Chebyshev polynomial method.
The black hole mass is always one.
"""
function eig_vals_vecs(
    nr::Integer,
    s::Integer,
    m::Integer,
    a::T,
    eps::T,
    om::Complex{T},
) where {T<:Real}

    bhm = T(1) ## always have unit black hole mass
    rmin = T(0) ## location of future null infinity (1/r = ∞)
    #rmax = abs(a) > 0 ? (bhm / (a^2)) * (1 - sqrt(1 - ((a / bhm)^2))) : 0.5 / bhm
    rp =  bhm + sqrt(bhm^2 - a^2)
    re = rp * (1 + eps)
    rmax = 1/re 
    LHS, RHS = radial_discretized_eqn_a_bordered(nr, s, m, a, bhm, om, rmin, rmax)
    t = eigen(LHS, RHS, sortby = abs)

    return -t.values[1], t.vectors[:, 1], CH.cheb_pts(rmin, rmax, nr)
end


end 