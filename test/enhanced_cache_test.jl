using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS
using OrdinaryDiffEqLowOrderRK
using DiffEqBase: ODEFunction, ODEProblem, solve
using SciMLBase: NullParameters, init, ReturnCode
using Test

@testset "Enhanced Cache Tests" begin
    # Test 1: Basic cache structure
    @testset "Cache has required fields" begin
        function ode1(du, u, p, t)
            @. du = -0.1u
        end
        f1 = ODEFunction(ode1)

        function ode2(du, u, p, t)
            du[1] = -0.01u[2]
            du[2] = -0.01u[1]
        end
        f2 = ODEFunction(ode2)

        u0 = [0.5, 0.3]
        tspan = (0.0, 10.0)

        split_f = OS.GenericSplitFunction((f1, f2), ([1, 2], [1, 2]))
        prob = OS.OperatorSplittingProblem(split_f, u0, tspan, (NullParameters(), NullParameters()))

        alg = OS.LieTrotterGodunov((Euler(), Euler()))
        integrator = init(prob, alg; dt=0.1)

        cache = integrator.cache
        
        # Check that all fields are present
        @test hasfield(typeof(cache), :u)
        @test hasfield(typeof(cache), :uprev)
        @test hasfield(typeof(cache), :t)
        @test hasfield(typeof(cache), :tprev)
        @test hasfield(typeof(cache), :dt)
        @test hasfield(typeof(cache), :dtcache)
        @test hasfield(typeof(cache), :sol)
        @test hasfield(typeof(cache), :controller)
        @test hasfield(typeof(cache), :EEst)
        @test hasfield(typeof(cache), :iter)
        @test hasfield(typeof(cache), :stats)
        @test hasfield(typeof(cache), :subintegrator_tree)
        @test hasfield(typeof(cache), :solution_index_tree)
        @test hasfield(typeof(cache), :synchronizer_tree)
        
        # Check sol is OperatorSplittingMinimalSolution
        @test cache.sol isa OS.OperatorSplittingMinimalSolution
        @test cache.sol.retcode == ReturnCode.Default
        
        # Check stats is IntegratorStats
        @test cache.stats isa OS.IntegratorStats
    end

    # Test 2: Nested splitting with enhanced cache as subintegrator
    @testset "Nested splitting structure" begin
        function ode1(du, u, p, t)
            du[1] = -0.1u[1]
        end
        f1 = ODEFunction(ode1)

        function ode2(du, u, p, t)
            du[1] = -0.01u[1]
        end
        f2 = ODEFunction(ode2)

        function ode3(du, u, p, t)
            du[1] = -0.005u[1]
        end
        f3 = ODEFunction(ode3)

        u0 = [0.5]
        tspan = (0.0, 10.0)

        # Create nested split
        inner_split = OS.GenericSplitFunction((f1, f2), ([1], [1]))
        outer_split = OS.GenericSplitFunction((inner_split, f3), ([1], [1]))

        prob = OS.OperatorSplittingProblem(outer_split, u0, tspan, 
            ((NullParameters(), NullParameters()), NullParameters()))

        inner_alg = OS.LieTrotterGodunov((Euler(), Euler()))
        alg = OS.LieTrotterGodunov((inner_alg, Euler()))
        integrator = init(prob, alg; dt=0.1)

        # Check that first subintegrator is an enhanced cache
        first_subint = integrator.subintegrator_tree[1]
        @test first_subint isa OS.AbstractOperatorSplittingCache
        @test hasfield(typeof(first_subint), :subintegrator_tree)
        @test hasfield(typeof(first_subint), :sol)
        @test first_subint.sol isa OS.OperatorSplittingMinimalSolution
    end

    # Test 3: Basic solve works
    @testset "Basic solve" begin
        function ode1(du, u, p, t)
            du[1] = -0.1*u[1]
            du[2] = -0.1*u[2]
        end
        f1 = ODEFunction(ode1)

        function ode2(du, u, p, t)
            du[1] = -0.01*u[2]
            du[2] = -0.01*u[1]
        end
        f2 = ODEFunction(ode2)

        u0 = [1.0, 1.0]
        tspan = (0.0, 1.0)

        split_f = OS.GenericSplitFunction((f1, f2), ([1, 2], [1, 2]))
        prob = OS.OperatorSplittingProblem(split_f, u0, tspan, (NullParameters(), NullParameters()))

        alg = OS.LieTrotterGodunov((Euler(), Euler()))
        sol = solve(prob, alg; dt=0.1)

        @test sol.retcode == ReturnCode.Success
    end
end
