################################################################################
# Energy broadening
################################################################################

function nonstationary_gaussian(instrument::DirectGeometrySpec; intrinsic_width=0)
    (; Ei, L1, L2, L3, Δtp, Δtc, Δtd) = instrument

    FWHM_to_sigma = 1/2sqrt(2log(2))

    # Calculate the resolution kernel at some representative points.
    Es = range(-Ei, Ei, 200)
    dEs = [FWHM_to_sigma*energy_resolution(Ei, E, L1, L2, L3, Δtp, Δtc, Δtd) for E in Es]
    dEs = map(dEs) do dE
        sqrt(dE^2 + intrinsic_width^2)
    end

    # First a third-order polynomial to resulting values to determine a FWHM function.
    @. poly3(x, p) = p[1]*x^3 + p[2]*x^2 + p[3]*x + p[4]
    fit = curve_fit(poly3, Es, dEs, 0.1*ones(4))
    sigma(E) = poly3(E, fit.param)

    return Sunny.NonstationaryBroadening((b, ω) -> exp(-(ω-b)^2/2sigma(b)^2) / √(2π*sigma(b)^2))
end
# σ = FWHM_to_sigma*SunnyAnalysisTools.energy_resolution(Ei, μ > Ei ? Ei : μ, L1, L2, L3, Δtp, Δtc, Δtd)
# σ_tot = sqrt(σ^2 + λ^2)


# This is a stub -- insert real model of triple-axis instrument later
function nonstationary_gaussian(instrument::TripleAxisSpec)
    (; name, params) = instrument

    sigma = if name == "SPINS"
        E -> 0.9*(0.0949 + 0.0182E) / √2
    elseif name == "HMI"
        if params["resolution"] == "high"
            E -> (0.0886 + 0.0015E) / √2
        elseif params["resolution"] == "low"
            E ->  max( (-0.0594 + 0.0981E) / √2, 0.1 / √2 )
        else
            error("Unknown resolution setting for HMI")
        end
    else
        error("Unknown triple-axis instrument")
    end

    return Sunny.NonstationaryBroadening((b, ω) -> exp(-(ω-b)^2/2sigma(b)^2) / √(2π*sigma(b)^2))
end


################################################################################
# Daniel's instrument model
################################################################################

# Takes meV, returns m/s
energy_to_velocity(E_meV) = sqrt(2 * E_meV * J_per_meV / mₙ)

# Takes m/s, returns m (recall h has units J*s = kg*m*s^-1)
velocity_to_wavelength(v_mps) = h / (v_mps * mₙ)

# Input units: meV, meV, inverse angstrom
function theta(Ei_meV, E_meV, Q)
    clip(val, lo, hi) = min(max(val, lo), hi) 

    Ef_meV = Ei_meV - E_meV
    vi = energy_to_velocity(Ei_meV)
    vf = energy_to_velocity(Ef_meV)
    Q = Q * angstrom_per_meter 

    cos_2theta = (vi^2 + vf^2 - (Q * ħ/mₙ)^2) / (2 * vi * vf)
    if !(-1 <= cos_2theta <= 1)
        @warn "Cos(2θ) not between -1 and 1! Resulted in clipping."
        cos_2theta = clip(cos_2theta, -1., 1.)
    end

    return acos(cos_2theta)/2
end

function energy_resolution(Ei_meV, E_meV, L1, L2, L3, Δtp, Δtc, Δtd)
    Ef_meV = Ei_meV - E_meV
    vi = energy_to_velocity(Ei_meV)
    vf = energy_to_velocity(Ef_meV)
    return mₙ * sqrt(
        ( (vi^3) + (vf^3)*L2/L3      )^2 * (Δtp/L1)^2 +
        ( (vi^3) + (vf^3)*(L1+L2)/L3 )^2 * (Δtc/L1)^2 +
        ( (vf^3) )^2 * (Δtd/L3)^2
    ) / J_per_meV
end


# meV (Es), inverse Angstrom (Q), meters (Ls). Return inverse angstrom
function dQx(Ei, E, Q, L1, L2, L3, Δtp, Δtd, Δtc, Δθ)
    Ef = Ei - E
    vi = energy_to_velocity(Ei)
    vf = energy_to_velocity(Ef)
    θ = theta(Ei, E, Q)

    return (mₙ/ħ) * sqrt(
        ( vi^2 + vf^2 * (L2/L3) * cos(2θ)      )^2 * (Δtp/L1)^2 +
        ( vi^2 + vf^2 * ((L1+L2)/L3) * cos(2θ) )^2 * (Δtc/L1)^2 +
        ( vf^2 * cos(2θ) )^2 * (Δtd/L3)^2 +
        ( vf * sin(2θ) )^2 * Δθ^2 
    ) / angstrom_per_meter
end

function dQy(Ei, E, Q, L1, L2, L3, Δtp, Δtd, Δtc, Δθ)
    Ef = Ei - E
    vf = energy_to_velocity(Ef)
    θ = theta(Ei, E, Q)

    return (mₙ/ħ) * sqrt(
        ( vf^2 * (L2/L3) * sin(2θ)        )^2 * (Δtp/L1)^2 + 
        ( vf^2 * ((L1+L2)/L3) * sin(2θ)   )^2 * (Δtc/L1)^2 +
        ( vf^2 * sin(2θ) )^2 * (Δtd/L3)^2 + 
        ( vf * cos(2θ) * Δθ )^2
    ) / angstrom_per_meter
end

function dQ(Ei, E, Q, L1, L2, L3, Δtp, Δtd, Δtc, Δθ)
    Ef = Ei - E
    vi = energy_to_velocity(Ei)
    vf = energy_to_velocity(Ef)
    θ = theta(Ei, E, Q)

    dQx_val = dQx(Ei, E, Q, L1, L2, L3, Δtp, Δtd, Δtc, Δθ)
    dQy_val = dQy(Ei, E, Q, L1, L2, L3, Δtp, Δtd, Δtc, Δθ)

    Qx = (mₙ/ħ) * (vi - vf * cos(2θ)) / angstrom_per_meter
    Qy = (mₙ/ħ) * (-vf * sin(2θ)) / angstrom_per_meter

    return (1/Q) * sqrt((Qx*dQx_val)^2 + (Qy*dQy_val)^2)
end

# Probably eliminate
function reciprocal_lattice(a, b, c, α, β, γ)
    α, β, γ = [α, β, γ] .* (π/180)

    # Volume of the direct unit cell
    V = a * b * c * sqrt(
        1 - cos(α)^2 - cos(β)^2 - cos(γ)^2
        + 2 * cos(α) * cos(β) * cos(γ)
    )

    # Reciprocal lattice parameters
    a_star = (b * c * sin(α)) / V
    b_star = (a * c * sin(β)) / V
    c_star = (a * b * sin(γ)) / V

    return 2π * a_star, 2π * b_star, 2π * c_star  # Include 2π factor
end
