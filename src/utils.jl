# helper function for setting up min/max heaps for tstops and saveat
function tstops_and_saveat_heaps(t0, tf, tstops, saveat)
    FT = typeof(tf)
    ordering = tf > t0 ? DataStructures.FasterForward : DataStructures.FasterReverse

    # ensure that tstops includes tf and only has values ahead of t0
    tstops = [filter(t -> t0 < t < tf || tf < t < t0, tstops)..., tf]
    tstops = DataStructures.BinaryHeap{FT, ordering}(tstops)

    if isnothing(saveat)
        saveat = [t0, tf]
    elseif saveat isa Number
        saveat > zero(saveat) || error("saveat value must be positive")
        saveat = tf > t0 ? saveat : -saveat
        saveat = [t0:saveat:tf..., tf]
    else
        # We do not need to filter saveat like tstops because the saving
        # callback will ignore any times that are not between t0 and tf.
        saveat = collect(saveat)
    end
    saveat = DataStructures.BinaryHeap{FT, ordering}(saveat)

    return tstops, saveat
end

"""
    need_sync(a, b)

This function determines whether it is necessary to synchronize two objects with any solution information.
A possible reason when no syncronization is necessary might be that the vectors alias each other in memory.
"""
need_sync

need_sync(a::AbstractVector, b::AbstractVector) = true
need_sync(a::SubArray, b::AbstractVector) = a.parent !== b
need_sync(a::AbstractVector, b::SubArray) = a !== b.parent
need_sync(a::SubArray, b::SubArray) = a.parent !== b.parent

"""
    sync_vectors!(a, b)

Copies the information in object b into object a, if syncronization is necessary.
"""
function sync_vectors!(a, b)
    if need_sync(a, b) && a !== b
        a .= b
    end
end

"""
     forward_sync_subintegrator!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, solution_indices, sync)

This function is responsible of copying the solution and parameters of the outer integrator and the synchronized subintegrators with the information given into the inner integrator.
If the inner integrator is synchronized with other inner integrators using `sync`, the function `forward_sync_external!` shall be dispatched for `sync`.
The `sync` object is passed from the outside and is the main entry point to dispatch custom types on for parameter synchronization.
The `solution_indices` are global indices in the outer integrators solution vectors.
"""
function forward_sync_subintegrator!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DiffEqBase.DEIntegrator, solution_indices, sync)
    forward_sync_internal!(outer_integrator, inner_integrator, solution_indices)
    forward_sync_external!(outer_integrator, inner_integrator, sync)
end

"""
    backward_sync_subintegrator!(outer_integrator::OperatorSplittingIntegrator, inner_integrator::DiffEqBase.DEIntegrator, solution_indices, sync)

This function is responsible of copying the solution of the inner integrator back into outer integrator and the synchronized subintegrators.
If the inner integrator is synchronized with other inner integrators using `sync`, the function `backward_sync_external!` shall be dispatched for `sync`.
The `sync` object is passed from the outside and is the main entry point to dispatch custom types on for parameter synchronization.
The `solution_indices` are global indices in the outer integrators solution vectors.
"""
function backward_sync_subintegrator!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DiffEqBase.DEIntegrator, solution_indices, sync)
    backward_sync_internal!(outer_integrator, inner_integrator, solution_indices)
    backward_sync_external!(outer_integrator, inner_integrator, sync)
end

# This is a bit tricky, because per default the operator splitting integrators share their solution vector. However, there is also the case
# when part of the problem is on a different device (thing e.g. about operator A being on CPU and B being on GPU).
# This case should be handled with special synchronizers.
function forward_sync_internal!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::OperatorSplittingIntegrator, solution_indices)
    nothing
end
function backward_sync_internal!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::OperatorSplittingIntegrator, solution_indices)
    nothing
end

function forward_sync_internal!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DiffEqBase.DEIntegrator, solution_indices)
    @views uouter = outer_integrator.u[solution_indices]
    sync_vectors!(inner_integrator.uprev, uouter)
    sync_vectors!(inner_integrator.u, uouter)
    SciMLBase.u_modified!(inner_integrator, true)
end
function backward_sync_internal!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DiffEqBase.DEIntegrator, solution_indices)
    @views uouter = outer_integrator.u[solution_indices]
    sync_vectors!(uouter, inner_integrator.u)
end

# This is a noop, because operator splitting integrators do not have parameters
function forward_sync_external!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::OperatorSplittingIntegrator, sync::NoExternalSynchronization)
    nothing
end
function forward_sync_external!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DiffEqBase.DEIntegrator, sync)
    synchronize_solution_with_parameters!(outer_integrator, inner_integrator.p, sync)
end

function backward_sync_external!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::OperatorSplittingIntegrator, sync::NoExternalSynchronization)
    nothing
end
function backward_sync_external!(outer_integrator::OperatorSplittingIntegrator,
        inner_integrator::DiffEqBase.DEIntegrator, sync)
    synchronize_solution_with_parameters!(outer_integrator, inner_integrator.p, sync)
end

function synchronize_solution_with_parameters!(outer_integrator::OperatorSplittingIntegrator, p::Any, sync)
    error("Outer synchronizer not dispatched for parameter type $(typeof(p)).")
end

# If we encounter NullParameters, then we have the convention for the standard sync map that no external solution is necessary.
function synchronize_solution_with_parameters!(
        outer_integrator::OperatorSplittingIntegrator, p::DiffEqBase.NullParameters, sync)
    nothing
end

# Default convention is that the first parameter serves as a buffer for the external solution
# function synchronize_solution_with_parameters!(outer_integrator::OperatorSplittingIntegrator, p::Tuple, sync)
#     @views uouter = outer_integrator.u[sync.parameter_indices]
#     sync_vectors!(p[1], uouter)
# end

# TODO this should go into a custom tree data structure instead of into a tuple-tree
function build_solution_index_tree(f::GenericSplitFunction)
    return ntuple(
        i->build_solution_index_tree_recursion(f.functions[i], f.solution_indices[i]),
        length(f.functions))
end

function build_solution_index_tree_recursion(f::GenericSplitFunction, solution_indices)
    return ntuple(
        i->build_solution_index_tree_recursion(f.functions[i], f.solution_indices[i]),
        length(f.functions))
end

function build_solution_index_tree_recursion(f, solution_indices)
    return solution_indices
end

function build_synchronizer_tree(f::GenericSplitFunction)
    return ntuple(i->build_synchronizer_tree_recursion(f.functions[i], f.synchronizers[i]), length(f.functions))
end

function build_synchronizer_tree_recursion(f::GenericSplitFunction, synchronizers)
    return ntuple(i->build_synchronizer_tree_recursion(f.functions[i], f.synchronizers[i]), length(f.functions))
end

function build_synchronizer_tree_recursion(f, synchronizer)
    return synchronizer
end
