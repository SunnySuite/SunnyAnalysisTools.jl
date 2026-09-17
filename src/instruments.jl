abstract type AbstractInstrumentSpec end

# abstract type TripleAxisSpec <: AbstractInstrumentSpec end

struct DirectGeometrySpec <: AbstractInstrumentSpec
    name :: String
    Ei   :: Float64
    L1   :: Float64
    L2   :: Float64
    L3   :: Float64
    Δtp  :: Float64
    Δtc  :: Float64
    Δtd  :: Float64
    Δθ   :: Float64
end

function Base.show(io::IO, obs::DirectGeometrySpec)
    (; name, Ei) = obs
    printstyled(io, "Direct Geometry Instrument\n"; bold=true, color=:underline)
    println(io, "Instrument name: ", name)
    println(io, "Incident energy: $Ei meV")
end

struct TripleAxisSpec <: AbstractInstrumentSpec
    name :: String
    # # instrument data from
    # # instrument configuration
    # # function wrapper to query TAVI
    # Precalculated resolution matrices
    params :: Dict{String, Any}
end

function Base.show(io::IO, obs::TripleAxisSpec)
    (; name) = obs
    printstyled(io, "Triple-axis Instrument\n"; bold=true, color=:underline)
    println(io, "Instrument name: ", name)
end

################################################################################
# Local reproduction of the PyChop (https://github.com/mantidproject/PyChop)
# calculations needed to build a DirectGeometrySpec for CNCS/HYSPEC/SEQUOIA/ARCS,
# so that no working PythonCall/PyChop install is required. Instrument
# parameters are read from the yaml files copied into data/pychop/ (themselves
# copies of PyChop's own instrument files); the physics below ports the
# relevant pieces of PyChop's Chop.py and MulpyRep.py.
################################################################################

const _PYCHOP_DATA_DIR = joinpath(@__DIR__, "..", "data", "pychop")
const _pychop_yaml_cache = Dict{String, Any}()

function _pychop_data(name)
    get!(_pychop_yaml_cache, name) do
        YAML.load_file(joinpath(_PYCHOP_DATA_DIR, name * ".yaml"))
    end
end

function _check_option(chosen, options, label)
    if isnothing(chosen) || !(chosen in options)
        valid = join(sort(String.(collect(options))), ", ")
        throw(ArgumentError("$label $(repr(chosen)) not recognized. Valid options: $valid"))
    end
end

# Constants and formulas below are direct ports of the relevant pieces of
# PyChop: Chop.tchop, Chop.tikeda, Moderator.getWidth (measured branch), and
# the disk-chopper opening-time calculation from MulpyRep.calcChopTimes.
const _PYCHOP_E2L = 81.8042103582802156   # meV*Å^2, so λ[Å] = sqrt(E2L/Ei[meV])
const _PYCHOP_SIGMA2FWHM = 2*sqrt(2*log(2))

# Fermi chopper time variance (s^2). freq: Hz, Ei: meV, pslit/radius/rho: m.
function _fermi_chopper_variance(freq, Ei, pslit, radius, rho)
    veloc = 437.3920 * sqrt(Ei)
    w = freq * 2π
    gamm = (2 * radius^2 / pslit) * abs(1/rho - 2*w/veloc)
    gamm >= 4 && return NaN
    gsqr = if gamm <= 1
        (1 - gamm^4/10) / (1 - gamm^2/6)
    else
        groot = sqrt(gamm)
        0.6 * gamm * (groot - 2)^2 * (groot + 8) / (groot + 4)
    end
    return (pslit / (2*radius*w))^2 / 6 * gsqr
end

_fermi_chopper_fwhm(freq, Ei, pslit, radius, rho) =
    sqrt(_fermi_chopper_variance(freq, Ei, pslit, radius, rho)) * _PYCHOP_SIGMA2FWHM

# Disk chopper opening FWHM (s). PyChop takes half the chopper's full opening
# duration as its FWHM; that duration doesn't depend on Ei or chopper phase
# (phase realignment shifts the opening/closing edges equally), so it reduces
# to this closed form. freq: Hz; slot_width/guide_width/radius: any common
# length unit (they only ever appear as a ratio).
function _disk_chopper_fwhm(freq, slot_width, guide_width, radius, numDisk)
    chop_vel = 2π * radius * numDisk * freq
    return (slot_width + guide_width) / (2 * chop_vel)
end

# Ikeda-Carpenter moderator time variance (s^2).
function _ikeda_moderator_variance(S1, S2, B1, B2, Emod, Ei)
    sig = sqrt(S1^2 + S2^2 * 81.8048/Ei)
    A = 4.37392e-4 * sig * sqrt(Ei)
    B = Ei > 130.0 ? B2 : B1
    R = exp(-Ei/Emod)
    tausqr = 3/A^2 + R*(2-R)/B^2
    return tausqr * 1e-12
end

_ikeda_moderator_fwhm(S1, S2, B1, B2, Emod, Ei) =
    sqrt(_ikeda_moderator_variance(S1, S2, B1, B2, Emod, Ei)) * _PYCHOP_SIGMA2FWHM

# Moderator FWHM (s) from a measured wavelength(Å) -> width(μs) table, linearly
# interpolated (matching scipy's "slinear"). Falls back to 0 outside the
# table's shortest wavelength, matching PyChop's own degenerate fallback for
# instruments (CNCS/HYSPEC) whose polynomial moderator model is unset.
function _measured_moderator_fwhm(Ei, wavelengths, widths; isSigma=false)
    idx = sortperm(wavelengths)
    λs, ws = wavelengths[idx], widths[idx]
    λ = sqrt(_PYCHOP_E2L / Ei)
    λ < λs[1] && return 0.0
    λc = min(λ, λs[end])
    i = clamp(searchsortedlast(λs, λc), 1, length(λs) - 1)
    w_us = ws[i] + (ws[i+1] - ws[i]) * (λc - λs[i]) / (λs[i+1] - λs[i])
    w = w_us / 1e6
    return isSigma ? w * _PYCHOP_SIGMA2FWHM : w
end

function _moderator_fwhm(moderator, Ei)
    if haskey(moderator, "measured_width")
        mw = moderator["measured_width"]
        return _measured_moderator_fwhm(Ei, Float64.(mw["wavelength"]), Float64.(mw["width"]);
                                         isSigma=get(mw, "isSigma", false))
    elseif moderator["imod"] == 1
        S1, S2, B1, B2, Emod = Float64.(moderator["mod_pars"])
        return _ikeda_moderator_fwhm(S1, S2, B1, B2, Emod, Ei)
    else
        error("Unsupported moderator model (imod=$(moderator["imod"])) with no measured_width table")
    end
end

# Distances (mod->final chopper, chopper->sample, sample->detector, mod->first
# chopper), matching PyChop's ChopperSystem.getDistances. When there's only
# one chopper total (SEQUOIA/ARCS), there's no separate upstream chopper to
# truncate the moderator pulse before the resolution chopper, so `xm` is the
# moderator face itself (0) rather than the resolution chopper's own
# distance -- otherwise x0==xm and the L1/Δtp correction below would degenerate to zero.
function _pychop_distances(cs)
    choppers = cs["choppers"]
    x0 = Float64(choppers[end]["distance"])
    xm = length(choppers) > 1 ? Float64(choppers[1]["distance"]) : 0.0
    return (x0, Float64(cs["chop_sam"]), Float64(cs["sam_det"]), xm)
end

# The resolution-chopper's running frequency: frequency_matrix * frequencies
# + constant_frequencies, matching PyChop's ChopperSystem._long_frequency.
# `freq`, if given, overrides `default_frequencies` (same shape/units as that
# yaml field) -- this is how PyChop's setChopper(package, freq) lets the user
# pick the independent chopper frequency/frequencies instead of taking the
# instrument's default.
function _pychop_final_frequency(cs, freq=nothing)
    fm = cs["frequency_matrix"]
    fvals = isnothing(freq) ? Float64.(cs["default_frequencies"]) : Float64.(freq)
    f0 = haskey(cs, "constant_frequencies") ? Float64.(cs["constant_frequencies"]) : zeros(length(fm))
    row = Float64.(fm[end])
    return sum(row .* fvals) + f0[end]
end

# Assembles a DirectGeometrySpec from the distances/widths above, applying
# the same first-chopper pulse-truncation correction as the original
# PythonCall-based cncs() implementation (see _pychop_distances for how `xm`
# is chosen so this is a no-op when there's no upstream chopper).
function _direct_geometry_spec(name, Ei, Δθ_deg, x0, x1, x2, xm, Δtp_raw, Δtc)
    L1, L2, L3 = x0 - xm, x1, x2
    Δtp = Δtp_raw * (1 - xm/x0)
    return DirectGeometrySpec(name, Ei, L1, L2, L3, Δtp, Δtc, 0.0, Δθ_deg * π/180)
end

"""
    cncs(; Ei, variant, freq=nothing, Δθ=1.5)

Build a `DirectGeometrySpec` for the CNCS direct-geometry spectrometer at
SNS. `variant` selects the resolution disk chopper's slot width and must be
one of CNCS's defined variants (e.g. `"High Flux"`, `"Intermediate"`,
`"High Resolution"`); an informative error listing the valid names is raised
if it is omitted or not recognized. `freq` overrides the resolution disk
(Chopper 4) frequency in Hz (default: 300 Hz); an error is raised if it
exceeds the chopper's maximum frequency. CNCS's other independent frequency,
the Fermi chopper (Chopper 1), is left at its default since it does not
enter into the resolution calculation. `Δθ` is the beam angular divergence
in degrees.
"""
function cncs(; Ei, variant=nothing, freq=nothing, Δθ=1.5)
    data = _pychop_data("cncs")
    cs = data["chopper_system"]
    variants = get(cs, "variants", Dict())
    _check_option(variant, keys(variants), "CNCS chopper variant")

    if !isnothing(freq)
        maxfreq = get(cs, "max_frequencies", nothing)
        if !isnothing(maxfreq) && freq > maxfreq[1]
            throw(ArgumentError("CNCS resolution disk frequency $freq Hz exceeds maximum of $(maxfreq[1]) Hz"))
        end
    end

    x0, x1, x2, xm = _pychop_distances(cs)

    final = cs["choppers"][end]
    slot_width, guide_width, radius = final["slot_width"], final["guide_width"], final["radius"]
    numDisk = get(final, "isDouble", false) ? 2 : 1

    override = get(variants[variant], "choppers", nothing)
    if !isnothing(override) && !isnothing(override[end])
        slot_width = get(override[end], "slot_width", slot_width)
        guide_width = get(override[end], "guide_width", guide_width)
        radius = get(override[end], "radius", radius)
    end

    freqvec = isnothing(freq) ? nothing : [freq, Float64(cs["default_frequencies"][2])]
    resfreq = _pychop_final_frequency(cs, freqvec)
    Δtc = _disk_chopper_fwhm(resfreq, slot_width, guide_width, radius, numDisk)
    Δtp_raw = _moderator_fwhm(data["moderator"], Ei)

    return _direct_geometry_spec("CNCS", Ei, Δθ, x0, x1, x2, xm, Δtp_raw, Δtc)
end

# Shared by hyspec/sequoia/arcs, which are all single-Fermi-chopper direct
# geometry instruments differing only in their yaml data and available
# chopper packages. Each of these instruments has a single independent
# chopper frequency (the Fermi chopper's), so `freq`, if given, is a scalar
# in Hz overriding the instrument's default_frequencies entry.
function _fermi_instrument(name; Ei, package, freq=nothing, Δθ)
    data = _pychop_data(lowercase(name))
    cs = data["chopper_system"]
    packages = cs["choppers"][end]["packages"]
    _check_option(package, keys(packages), "$name Fermi chopper package")

    if !isnothing(freq)
        maxfreq = get(cs, "max_frequencies", nothing)
        if !isnothing(maxfreq) && freq > maxfreq[1]
            throw(ArgumentError("$name chopper frequency $freq Hz exceeds maximum of $(maxfreq[1]) Hz"))
        end
    end

    x0, x1, x2, xm = _pychop_distances(cs)
    freq = _pychop_final_frequency(cs, isnothing(freq) ? nothing : [freq])
    pkg = packages[package]
    Δtc = _fermi_chopper_fwhm(freq, Ei, pkg["pslit"]/1000, pkg["radius"]/1000, pkg["rho"]/1000)
    Δtp_raw = _moderator_fwhm(data["moderator"], Ei)

    return _direct_geometry_spec(name, Ei, Δθ, x0, x1, x2, xm, Δtp_raw, Δtc)
end

"""
    hyspec(; Ei, package, freq=nothing, Δθ=1.5)

Build a `DirectGeometrySpec` for the HYSPEC direct-geometry spectrometer at
SNS. `package` selects the Fermi chopper package and must be one of HYSPEC's
defined packages; an informative error listing the valid names is raised if
it is omitted or not recognized. `freq` overrides the Fermi chopper
frequency in Hz (default: HYSPEC's nominal 180 Hz); an error is raised if it
exceeds the chopper's maximum frequency. `Δθ` is the beam angular divergence
in degrees.
"""
hyspec(; Ei, package=nothing, freq=nothing, Δθ=1.5) = _fermi_instrument("HYSPEC"; Ei, package, freq, Δθ)

"""
    sequoia(; Ei, package, freq=nothing, Δθ=1.5)

Build a `DirectGeometrySpec` for the SEQUOIA direct-geometry spectrometer at
SNS. `package` selects the Fermi chopper package (e.g. `"Fine"`, `"Sloppy"`)
and must be one of SEQUOIA's defined packages; an informative error listing
the valid names is raised if it is omitted or not recognized. `freq`
overrides the Fermi chopper frequency in Hz (default: SEQUOIA's nominal 300
Hz); an error is raised if it exceeds the chopper's maximum frequency. `Δθ`
is the beam angular divergence in degrees.
"""
sequoia(; Ei, package=nothing, freq=nothing, Δθ=1.5) = _fermi_instrument("SEQUOIA"; Ei, package, freq, Δθ)

"""
    arcs(; Ei, package, freq=nothing, Δθ=1.5)

Build a `DirectGeometrySpec` for the ARCS direct-geometry spectrometer at
SNS. `package` selects the Fermi chopper package and must be one of ARCS's
defined packages; an informative error listing the valid names is raised if
it is omitted or not recognized. `freq` overrides the Fermi chopper
frequency in Hz (default: ARCS's nominal 300 Hz); an error is raised if it
exceeds the chopper's maximum frequency. `Δθ` is the beam angular divergence
in degrees.
"""
arcs(; Ei, package=nothing, freq=nothing, Δθ=1.5) = _fermi_instrument("ARCS"; Ei, package, freq, Δθ)
