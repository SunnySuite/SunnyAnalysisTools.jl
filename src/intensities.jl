################################################################################
# Time-of-flight intensities calculations 
################################################################################
function apply_observation_nan_mask!(res, observation)
    if isnothing(observation)
        return res
    end

    @assert size(res) == size(observation.ints) "observation intensities must match calculated intensity dimensions"

    for i in eachindex(res)
        if isnan(observation.ints[i])
            res[i] = NaN
        end
    end

    return res
end

function calculate_intensities(swt::Sunny.AbstractSpinWaveTheory, broadening_spec::StationaryQConvolution;
    params=nothing,
    observation = nothing,
    kwargs...
)
    (; qpoints, epoints, qidcs, eidcs, qkernel, ekernel, binning) = broadening_spec
    (; qcenters, Es) = binning

    # Calculate intensities for all points in subsuming grid around bin.
    res = Sunny.intensities(swt, qpoints[:]; energies=epoints, kernel=ekernel, kwargs...)
    data = reshape(res.data, (length(epoints), size(qpoints)...))

    # Convolve along Q-axes only using an FFT. Unfortunately, energy is the fast
    # axis. `ifft(fft(x).*fft(h))` is already the correctly-normalized discrete
    # circular convolution (qkernel is itself normalized to sum to 1, see
    # StationaryQConvolution) -- no extra division needed here.
    data_ft = fft(data, (2, 3, 4))
    for i in axes(data_ft, 1)
        data_ft[i,:,:,:] .*= qkernel
    end
    data_conv = real.(ifft(data_ft, (2, 3, 4)))

    # Sum over samples that lie within each bin and normalize by number of
    # samples. The result is the bin average (an intensity density), not
    # multiplied by bin volume, so it is directly comparable to MDNorm/Shiver
    # histograms across cuts with different binnings.
    res = zeros(length(Es), size(qcenters)...)
    for i in CartesianIndices(qcenters), j in eachindex(Es)
        for (ei, qi) in Iterators.product(eidcs[j], qidcs[i])
            res[j, i] += data_conv[ei,qi]
        end
        res[j, i] /= length(eidcs[j]) * length(qidcs[i])
    end

    apply_observation_nan_mask!(res, observation)

    # spec and params
    ModelCalculation(res, binning, broadening_spec, params)
end


function calculate_intensities(swt::Sunny.AbstractSpinWaveTheory, broadening_spec::UniformSampling;
    params=nothing,
    unit_intensity=false, 
    thresh=1e-12, 
    observation = nothing, 
    kwargs...
)
    (; qpoints, epoints, qidcs, eidcs, ekernel, binning) = broadening_spec
    (; qcenters, Es) = binning

    dispersion_and_intensities = Sunny.intensities_bands(swt, qpoints[:])
    if unit_intensity
        dispersion_and_intensities.data .= map(dispersion_and_intensities.data) do val
            val > thresh ? 1.0 : 0.0
        end
    end
    res = Sunny.broaden(dispersion_and_intensities; energies=epoints, kernel=ekernel, kwargs...)
    data = reshape(res.data, (length(epoints), size(qpoints)...))

    # Sum over samples that lie within each bin and normalize by number of
    # samples (bin average; see StationaryQConvolution).
    res = zeros(length(Es), size(qcenters)...)
    for i in CartesianIndices(qcenters), j in eachindex(Es)
        res[j, i] = accumulate_bin_average(data, eidcs[j], qidcs[i])
    end

    apply_observation_nan_mask!(res, observation)

    # spec and params
    ModelCalculation(res, binning, broadening_spec, params)
end

# Temporary function. Eventually make an abstract calculation type in Sunny to
# be able leverage Sunny tools.
function calculate_intensities_domains(swt::Sunny.AbstractSpinWaveTheory, broadening_spec::UniformSampling, rotations, weights;
    params=nothing,
    observation = nothing, 
    kwargs...
)
    (; qpoints, epoints, qidcs, eidcs, ekernel, binning) = broadening_spec
    (; qcenters, Es) = binning

    R0, Rs... = Sunny.rotation_in_rlu.(Ref(binning.crystal), rotations)
    w0, ws... = weights

    dispersion_and_intensities = Sunny.intensities_bands(swt, map(q -> R0*q, qpoints[:]))
    sunnyres = Sunny.broaden(dispersion_and_intensities; energies=epoints, kernel=ekernel, kwargs...)
    data = reshape(sunnyres.data, (length(epoints), size(qpoints)...))
    res = zeros(length(Es), size(qcenters)...)
    for i in CartesianIndices(qcenters), j in eachindex(Es)
        res[j, i] = accumulate_bin_average(data, eidcs[j], qidcs[i])
    end
    res .*= w0

    for (R, w) in zip(Rs, ws)
        dispersion_and_intensities = Sunny.intensities_bands(swt, map(q -> R*q, qpoints[:]))
        sunnyres = Sunny.broaden(dispersion_and_intensities; energies=epoints, kernel=ekernel, kwargs...)
        data = reshape(sunnyres.data, (length(epoints), size(qpoints)...))
        res_loc = zeros(length(Es), size(qcenters)...)
        for i in CartesianIndices(qcenters), j in eachindex(Es)
            res_loc[j, i] = accumulate_bin_average(data, eidcs[j], qidcs[i])
        end
        res_loc .*= w
        res .+= res_loc
    end

    apply_observation_nan_mask!(res, observation)

    # spec and params
    ModelCalculation(res, binning, broadening_spec, params)
end


accumulate_bin_average(data, einds, qinds) = sum(data[ei, qi] for (ei, qi) in Iterators.product(einds, qinds)) / (length(einds) * length(qinds))

uniform_bin_samples(Ecenter, ΔE, nepoints) = [Ecenter - ΔE/2 + (i - 0.5) * ΔE / nepoints for i in 1:nepoints]


function calculate_intensities(swt::Sunny.AbstractSpinWaveTheory, broadening_spec::LatinHyperCube;
    params=nothing,
    unit_intensity=false,
    thresh=1e-12,
    observation = nothing,
    kwargs...
)
    (; binning, nqpoints, nepoints, rng, ekernel) = broadening_spec
    (; qcenters, Es, directions, Δs) = binning

    bounds = [(-Δ/2, Δ/2) for Δ in Δs[1:3]]
    ΔE = Δs[4]

    # Sample Q and E locally within each bin using Latin hypercubes.
    # Q samples and their dispersions are shared across all energy bins at a
    # fixed q-bin to avoid repeating Sunny.intensities_bands work.
    res = zeros(length(Es), size(qcenters)...)
    for i in CartesianIndices(qcenters)
        qcenter = SVector{3, Float64}(qcenters[i]...)
        qsamples = latin_hypercube_points(qcenter, directions, bounds, nqpoints; rng)

        dispersion_and_intensities = Sunny.intensities_bands(swt, qsamples)
        if unit_intensity
            dispersion_and_intensities.data .= map(dispersion_and_intensities.data) do val
                val > thresh ? 1.0 : 0.0
            end
        end

        # Build all per-energy-bin uniform samples for this q-bin,
        # then issue a single broaden call with globally sorted energies.
        nEs = length(Es)
        all_esamples_unsorted = Vector{Float64}(undef, nepoints * nEs)
        for j in eachindex(Es)
            offset = (j - 1) * nepoints
            esamples_j = uniform_bin_samples(Es[j], ΔE, nepoints)
            all_esamples_unsorted[offset+1:offset+nepoints] = esamples_j
        end

        perm = sortperm(all_esamples_unsorted)
        all_esamples = all_esamples_unsorted[perm]
        row_for_flat = invperm(perm)

        broadened = Sunny.broaden(dispersion_and_intensities; energies=all_esamples, kernel=ekernel, kwargs...)
        data = reshape(broadened.data, (length(all_esamples), length(qsamples)))

        for j in eachindex(Es)
            offset = (j - 1) * nepoints
            erows = row_for_flat[offset+1:offset+nepoints]
            res[j, i] = accumulate_bin_average(data, erows, eachindex(qsamples))
        end
    end

    apply_observation_nan_mask!(res, observation)

    ModelCalculation(res, binning, broadening_spec, params)
end


################################################################################
# TAX Intensities Functions
################################################################################

# For a single HKLE point and resolution kernel, calculate the convolved
# intensity using a Sunny spin wave theory (swt). Sums over a grid of intensities
# at neihboring HKLs about the given point, with the intensities weighted by the
# convolution kernel. No effort here is made to normalize (i.e., there is no
# differential element, ΔHΔKΔLΔE, included in the sum.)
function tax_convolved_intensity_grid(intfunc, qe0, K, directions, bounds, counts)
    qh, qk, ql, _ = qe0

    HKLs = grid_points(SVector{3, Float64}(qh, qk, ql), directions, bounds, counts)
    HKLs = reshape(HKLs, length(HKLs))  # Interpret as linear array so intensities_bands will accept it

    (; data, disp) = intfunc(HKLs)

    cumval = 0.0
    for (iq, q) in enumerate(HKLs), iband in axes(disp, 1)
        qe = SVector{4, Float64}(q..., disp[iband, iq])
        cumval += data[iband, iq] * gaussian_func(qe, qe0, K)
    end

    return cumval # Multiply by differential when considering absolute units
end


function principal_axes_of_gaussian(Σ)
    vals, vecs = eigen(Σ)
    σs = sqrt.(vals)
    [σ*axis for (σ, axis) in zip(σs, eachcol(vecs))]
end

# Get the principal axes of the distribution as well as their relative 
# magnitudes in multiples of the corresponding eigenvalues corresponding 
# the principal axes. This is useful for defining a bounding box for the 
# grid of sampled qs.
function directions_and_bounds(Σ; nsigmas=3)
    vals, directions = eigen(Σ)
    σs = sqrt.(vals)
    bounds = [nsigmas .* (-σ, σ) for σ in σs]
    return (; directions, bounds)
end

function calculate_intensities(swt::Sunny.AbstractSpinWaveTheory, taxspec::TripleAxisGrid{2}; kwargs...)
    (; path, Ks, nsigmas, counts) = taxspec
    (; HKLs, Es, projection) = path
    buf = zeros(length(HKLs), length(path.Es))
    intfunc(hkls) = Sunny.intensities_bands(swt, hkls; kwargs...)
    for (n, ((j, HKL), (k, E))) in enumerate(Iterators.product(enumerate(HKLs), enumerate(Es)))
        K = Ks[n]
        q = projection*HKL
        (; directions, bounds) = directions_and_bounds(inv(K); nsigmas)
        directions = directions[1:3, 1:3]
        bounds = bounds[1:3]
        buf[j, k] = tax_convolved_intensity_grid(intfunc, SVector{4, Float64}(q..., E), K, directions, bounds, counts)  
    end
    return buf
end

function calculate_intensities(intfunc::Function, taxspec::TripleAxisGrid{2}; kwargs...)
    (; path, Ks, nsigmas, counts) = taxspec
    (; HKLs, Es, projection) = path
    buf = zeros(length(HKLs), length(path.Es))
    for (n, ((j, HKL), (k, E))) in enumerate(Iterators.product(enumerate(HKLs), enumerate(Es)))
        K = Ks[n]
        q = projection*HKL
        (; directions, bounds) = directions_and_bounds(inv(K); nsigmas)
        directions = directions[1:3, 1:3]
        bounds = bounds[1:3]
        buf[j, k] = tax_convolved_intensity_grid(intfunc, SVector{4, Float64}(q..., E), K, directions, bounds, counts)  
    end
    return buf
end

function calculate_intensities(intfunc::Function, path, Ks, nsigmas, counts; kwargs...)
    # (; path, Ks, nsigmas, counts) = taxspec
    (; HKLs, Es, projection) = path
    buf = zeros(length(HKLs), length(path.Es))
    for (n, ((j, HKL), (k, E))) in enumerate(Iterators.product(enumerate(HKLs), enumerate(Es)))
        K = Ks[n]
        q = projection*HKL
        (; directions, bounds) = directions_and_bounds(inv(K); nsigmas)
        directions = directions[1:3, 1:3]
        bounds = bounds[1:3]
        buf[j, k] = tax_convolved_intensity_grid(intfunc, SVector{4, Float64}(q..., E), K, directions, bounds, counts)  
    end
    return buf
end

function tax_convolved_intensity_mc(intfunc, qe0, K, nsamps)
    Σ = inv(K)
    qes = sample_q(qe0, Σ, nsamps)
    hkls = [Sunny.Vec3(qe[1], qe[2], qe[3]) for qe in eachcol(qes)]
    hkls = reshape(hkls, length(hkls))

    # (; data, disp) = intensities_bands(swt, hkls)
    (; data, disp) = intfunc(hkls)

    cumval = 0.0
    for (iq, q) in enumerate(hkls), iband in axes(disp, 1)
        qe = SVector{4, Float64}(q..., disp[iband, iq])
        cumval += data[iband, iq] * gaussian_func(qe, qe0, K) # Shouldn't have to multiply by gaussian_func...
        # cumval += data[iband, iq] 
    end

    return cumval/nsamps 
end

function calculate_intensities(swt::Sunny.AbstractSpinWaveTheory, taxspec::TripleAxisMC{1}; kwargs...)
    (; path, N, Ks) = taxspec 
    (; HKLs, Es, projection) = path

    buf = zero(path.Es)
    intfunc(hkls) = Sunny.intensities_bands(swt, hkls; kwargs...)
    for (n, (HKL, K, E)) in enumerate(zip(HKLs, Ks, Es))
        q = projection*HKL
        buf[n] = tax_convolved_intensity_mc(intfunc, SVector{4, Float64}(q..., E), K, N)
    end
    return buf
end

function calculate_intensities(swt::Sunny.AbstractSpinWaveTheory, taxspec::TripleAxisMC{2}; kwargs...)
    (; path, N, Ks) = taxspec 
    (; HKLs, Es, projection) = path

    buf = zeros(length(HKLs), length(path.Es))
    intfunc(hkls) = Sunny.intensities_bands(swt, hkls; kwargs...)
    for (n, ((j, HKL), (k, E))) in enumerate(Iterators.product(enumerate(HKLs), enumerate(Es)))
        K = Ks[n]
        q = projection*HKL
        buf[j,k] = tax_convolved_intensity_mc(intfunc, SVector{4, Float64}(q..., E), K, N)
    end
    return buf
end

# function calculate_intensities(intfunc::Function, path, Ks, nsigmas, counts; kwargs...)
function calculate_intensities(intfunc::Function, path, Ks, N; kwargs...)
    # (; path, N, Ks) = taxspec 
    (; HKLs, Es, projection) = path

    buf = zeros(length(HKLs), length(path.Es))
    for (n, ((j, HKL), (k, E))) in enumerate(Iterators.product(enumerate(HKLs), enumerate(Es)))
        K = Ks[n]
        q = projection*HKL
        buf[j,k] = tax_convolved_intensity_mc(intfunc, SVector{4, Float64}(q..., E), K, N)
    end
    return buf
end