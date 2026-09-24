using SunnyAnalysisTools
using Sunny
using StaticArrays
using Random
using LinearAlgebra
using Test

@testset "SunnyAnalysisTools.jl" begin
    @testset "LatinHyperCube sampling" begin
        qcenter = SVector{3, Float64}(0.0, 0.0, 0.0)
        directions = Matrix{Float64}(I, 3, 3)
        bounds = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)]
        rng = MersenneTwister(42)
        points = SunnyAnalysisTools.latin_hypercube_points(qcenter, directions, bounds, 5; rng)

        @test length(points) == 5

        strata = [Int[] for _ in 1:3]
        for p in points
            local_coords = inv(directions) * (p - qcenter)
            for d in 1:3
                lo, hi = bounds[d]
                t = (local_coords[d] - lo) / (hi - lo)
                push!(strata[d], floor(Int, t * 5) + 1)
            end
        end

        for d in 1:3
            @test sort(strata[d]) == collect(1:5)
        end
    end

    @testset "LatinHyperCube constructor" begin
        crystal = Sunny.Crystal(Sunny.lattice_vectors(5.0, 5.0, 5.0, 90, 90, 90), [[0.0, 0.0, 0.0]])
        directions = Matrix{Float64}(I, 3, 3)
        binning = UniformBinning(crystal, directions, [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        instrument = DirectGeometrySpec("test", 5.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        ekernel = nonstationary_gaussian(instrument)
        spec = LatinHyperCube(binning, ekernel; nqpoints=4, nepoints=3, rng=MersenneTwister(1))

        @test spec isa LatinHyperCube
        @test spec.nqpoints == 4
        @test spec.nepoints == 3
        @test spec.rng isa AbstractRNG
    end

    @testset "Bin intensities are bin averages (independent of bin volume)" begin
        # Ferromagnetic chain: smooth dispersion along a*.
        crystal = Sunny.Crystal(Sunny.lattice_vectors(3.0, 8.0, 8.0, 90, 90, 90), [[0.0, 0.0, 0.0]])
        sys = Sunny.System(crystal, [1 => Sunny.Moment(s=1, g=2)], :dipole)
        Sunny.set_exchange!(sys, -1.0, Sunny.Bond(1, 1, [1, 0, 0]))
        Sunny.randomize_spins!(sys)
        Sunny.minimize_energy!(sys)
        swt = Sunny.SpinWaveTheory(sys; measure=Sunny.ssf_perp(sys))
        ekernel = Sunny.gaussian(fwhm=1.0)
        directions = Matrix{Float64}(I, 3, 3)

        # One coarse bin sampled 2x per axis hits exactly the same points as
        # the 2^4 fine bins (sampled 1x per axis) that tile it, so the coarse
        # value must equal the mean of the fine values.
        coarse = UniformBinning(crystal, directions, [0.2], [0.1], [0.1], [1.5], [0.2, 0.2, 0.2, 1.0])
        fine = UniformBinning(crystal, directions, [0.15, 0.25], [0.05, 0.15], [0.05, 0.15], [1.25, 1.75], [0.1, 0.1, 0.1, 0.5])

        calc_coarse = calculate_intensities(swt, UniformSampling(coarse, ekernel; nperqbin=2, nperebin=2))
        calc_fine = calculate_intensities(swt, UniformSampling(fine, ekernel; nperqbin=1, nperebin=1))
        @test size(calc_fine.data) == (2, 2, 2, 2)
        @test calc_coarse.data[1] ≈ sum(calc_fine.data) / length(calc_fine.data)
    end

    @testset "Direct geometry instruments" begin
        for (f, kwargs) in [(cncs, (variant="High Flux",)), (hyspec, (package="OnlyOne",)),
                             (sequoia, (package="High-Flux",)), (arcs, (package="ARCS-100-1.5-AST",))]
            spec = f(; Ei=20.0, kwargs...)
            @test spec isa DirectGeometrySpec
            for field in (:L1, :L2, :L3, :Δtp, :Δtc, :Δθ)
                @test isfinite(getfield(spec, field))
                @test getfield(spec, field) > 0
            end
            ekernel = nonstationary_gaussian(spec)
            @test ekernel isa Sunny.NonstationaryBroadening
        end

        @test_throws "Valid options" cncs(; Ei=20.0)
        @test_throws "Valid options" cncs(; Ei=20.0, variant="not a variant")
        @test_throws "Valid options" hyspec(; Ei=20.0)
        @test_throws "Valid options" sequoia(; Ei=20.0, package="not a package")
        @test_throws "Valid options" arcs(; Ei=20.0, package="not a package")
    end

    @testset "read_shiver_ascii" begin
        file = joinpath(@__DIR__, "..", "reference_material", "ascii_slice.dat")
        obs = read_shiver_ascii(file)

        @test obs isa TimeOfFlightObservation
        @test size(obs.ints) == (72, 75, 1, 1)
        @test size(obs.errs) == (72, 75, 1, 1)
        @test obs.binning.labels == ["DeltaE", "[H,H,0]", "[0,0,L]", "[H,-H,0]"]

        # 75 bins from 0.0 to 0.75, step 0.01, along [H,H,0]
        @test length(obs.binning.Us) == 75
        @test isapprox(obs.binning.Us[1], 0.005; atol=1e-10)
        @test isapprox(obs.binning.Us[end], 0.745; atol=1e-10)
        @test isapprox(obs.binning.Δs[1], 0.01; atol=1e-10)

        # 72 bins from 0.2 to 2.0, step 0.025, along DeltaE
        @test length(obs.binning.Es) == 72
        @test isapprox(obs.binning.Es[1], 0.2125; atol=1e-10)
        @test isapprox(obs.binning.Es[end], 1.9875; atol=1e-10)
        @test isapprox(obs.binning.Δs[4], 0.025; atol=1e-10)

        # Singleton axes collapse to their midpoint, with width = hi - lo
        @test obs.binning.Vs ≈ [-3.0]
        @test isapprox(obs.binning.Δs[2], 0.1; atol=1e-10)
        @test obs.binning.Ws ≈ [0.0]
        @test isapprox(obs.binning.Δs[3], 0.2; atol=1e-10)

        # directions parsed from the Binning: bracket labels [H,H,0], [0,0,L], [H,-H,0]
        @test obs.binning.directions ≈ [1.0 0.0 1.0; 1.0 0.0 -1.0; 0.0 1.0 0.0]

        # Lattice parameters parsed from UnitCell: (latvecs columns are a, b, c)
        @test isapprox(norm(obs.binning.crystal.latvecs[:,1]), 5.71082; atol=1e-3)
        @test isapprox(norm(obs.binning.crystal.latvecs[:,2]), 5.70969; atol=1e-3)
        @test isapprox(norm(obs.binning.crystal.latvecs[:,3]), 21.4314; atol=1e-3)

        # Metadata not yet structurally used (instrument info) is preserved, not discarded
        @test obs.params["name"] == "Histogram_1"
        @test obs.params["instrument_name"] == "CNCS"
        @test obs.params["Ei"] ≈ 2.7
        @test obs.params["orientation_u"] ≈ [0.76409, 0.91221, 20.4881]
        @test obs.params["orientation_v"] ≈ [2.70519, 2.75431, -6.28141]

        # Spot check specific rows against the raw file, to confirm the
        # reshape/permutation lands values at the correct (E,U,V,W) index.
        eidx = findfirst(e -> isapprox(e, 0.2375), obs.binning.Es)
        uidx = findfirst(u -> isapprox(u, 0.005), obs.binning.Us)
        @test isnan(obs.ints[eidx, uidx, 1, 1])  # "nan nan 5.0e-3 2.375e-1 -3.0 0.0"

        eidx2 = findfirst(e -> isapprox(e, 1.0375), obs.binning.Es)
        @test obs.ints[eidx2, uidx, 1, 1] == 0.0  # a genuine (non-NaN) zero-intensity row
        @test obs.errs[eidx2, uidx, 1, 1] == 0.0

        @test count(!isnan, obs.ints) == 3977
    end
end
