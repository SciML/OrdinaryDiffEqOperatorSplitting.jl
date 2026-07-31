using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS
using Test

import SciMLBase
import DiffEqBase

# Minimal stand-in for a leaf integrator whose buffers downstream packages may alias
# into the parent's buffers (e.g. views into the master solution for GPU setups).
mutable struct MockLeaf{U, UP} <: SciMLBase.AbstractODEIntegrator{Nothing, true, U, Float64}
    u::U
    uprev::UP
    u_modified::Bool
end

@static if isdefined(SciMLBase, :derivative_discontinuity!)
    SciMLBase.derivative_discontinuity!(m::MockLeaf, b) = m.u_modified = b
end
@static if isdefined(DiffEqBase, :u_modified!)
    DiffEqBase.u_modified!(m::MockLeaf, b) = m.u_modified = b
end

@testset "forward_sync_internal!" begin
    @testset "independent child buffers are synced" begin
        u_parent = [1.0, 2.0, 3.0]
        uprev_parent = [10.0, 20.0, 30.0]
        idxs = [1, 2]
        child = MockLeaf([0.0, 0.0], [0.0, 0.0], false)

        OS.forward_sync_internal!(u_parent, uprev_parent, child, idxs)
        @test child.u == u_parent[idxs]
        @test child.uprev == u_parent[idxs]
        @test child.u_modified
    end

    @testset "a child uprev aliasing the parent rollback buffer survives" begin
        # A child whose uprev is a view into the parent's uprev: the sync must not
        # scribble on the parent's rollback anchor mid-step.
        u_parent = [1.0, 2.0, 3.0]
        uprev_parent = [10.0, 20.0, 30.0]
        idxs = [1, 2]
        child = MockLeaf([0.0, 0.0], view(uprev_parent, idxs), false)

        OS.forward_sync_internal!(u_parent, uprev_parent, child, idxs)
        @test child.u == u_parent[idxs]
        @test uprev_parent == [10.0, 20.0, 30.0]  # untouched
        @test child.u_modified
    end
end
