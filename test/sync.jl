using OrdinaryDiffEqOperatorSplitting
import OrdinaryDiffEqOperatorSplitting as OS
using Test

import SciMLBase
import DiffEqBase: DiffEqBase, ODEFunction
using OrdinaryDiffEqLowOrderRK

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

@testset "need_sync" begin
    # A device array package's own contiguous view (e.g. GPUArrays' `derive`) comes
    # back as a *new* dense wrapper over the same buffer, not a `SubArray`. Emulate
    # that aliasing relationship on the host with `unsafe_wrap` over the same
    # pointer; `a` must outlive the wrap, hence `GC.@preserve`.
    @testset "aliasing dense vectors need no sync" begin
        a = [1.0, 2.0, 3.0]
        GC.@preserve a begin
            b = unsafe_wrap(Array, pointer(a), length(a))
            @test !OS.need_sync(a, b)
            @test !OS.need_sync(b, a)
        end
    end

    @testset "non-aliasing dense vectors need a sync" begin
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]  # equal content, distinct memory
        c = [4.0, 5.0, 6.0]  # distinct content and memory
        @test OS.need_sync(a, b)
        @test OS.need_sync(a, c)
    end

    @testset "SubArray cases are unchanged" begin
        parent = [1.0, 2.0, 3.0]
        other = [1.0, 2.0, 3.0]
        view_of_parent = view(parent, 1:2)

        @test !OS.need_sync(view_of_parent, parent)
        @test OS.need_sync(view_of_parent, other)
        @test !OS.need_sync(parent, view_of_parent)
        @test OS.need_sync(other, view_of_parent)
        @test !OS.need_sync(view(parent, 1:2), view(parent, 1:3)) # same parent
        @test OS.need_sync(view(parent, 1:2), view(other, 1:2))   # different parent
    end
end

@testset "Nested children address their parent's buffers" begin
    # dof indices are parent-relative at every level, so a grandchild's initial
    # state is parent-slice-of-parent-slice -- NOT the root vector indexed with
    # parent-relative indices. Stock (copying) leaves hide a mix-up because the
    # first forward sync overwrites their state; leaves wired as views into the
    # handed buffer would alias the wrong root slots permanently.
    ode1(du, u, p, t) = (du .= -0.1 .* u; nothing)
    ode2(du, u, p, t) = (du[1] = -0.01 * u[2]; du[2] = -0.01 * u[1]; nothing)
    u0 = [10.0, 20.0, 30.0]
    f1dofs = [1, 2, 3]
    f2dofs = [1, 3]
    f3dofs = [1, 2] # relative to the inner node's [1, 3] slice → root slots 1 and 3
    inner = GenericSplitFunction((ODEFunction(ode2), ODEFunction(ode2)), (f3dofs, f3dofs))
    outer = GenericSplitFunction((ODEFunction(ode1), inner), (f1dofs, f2dofs))
    prob = OperatorSplittingProblem(outer, u0, (0.0, 1.0))
    alg = LieTrotterGodunov((Euler(), LieTrotterGodunov((Euler(), Euler()))))
    integ = DiffEqBase.init(prob, alg; dt = 0.1, adaptive = false)

    sub = integ.child_subintegrators[2]
    @test sub.u == u0[f2dofs]                                 # [10, 30]
    @test sub.child_subintegrators[1].u == u0[f2dofs][f3dofs] # [10, 30], not [10, 20]
    @test sub.child_subintegrators[2].u == u0[f2dofs][f3dofs]
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
