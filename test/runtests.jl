using Test
using SafeTestsets

@safetestset "Operator Splitting API" include("operator_splitting_api.jl")
@safetestset "Aliasing" include("alias_u0.jl")
@safetestset "Consistency" include("consistency.jl")
 # TODO MTK test