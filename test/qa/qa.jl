using OrdinaryDiffEqOperatorSplitting
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(OrdinaryDiffEqOperatorSplitting)
end

@testset "JET" begin
    JET.test_package(OrdinaryDiffEqOperatorSplitting; target_defined_modules = true)
end
