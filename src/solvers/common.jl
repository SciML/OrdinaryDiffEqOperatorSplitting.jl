# ---------------------------------------------------------------------------
# Pieces shared by every splitting scheme
# ---------------------------------------------------------------------------

"""
    _advance_child!(parent, child, i, dt)

Advance the `i`-th operator of `parent` by `dt`, syncing state into the child before
and out of it afterwards. A failed child is reported through `parent.force_stepfail`,
which every scheme checks between flows.
"""
function _advance_child!(parent, child, i, dt)
    idxs = parent.child_solution_indices[i]
    sync = parent.child_synchronizers[i]

    @timeit_debug "sync ->" forward_sync_subintegrator!(parent, child, idxs, sync)
    @timeit_debug "time solve" advance_solution_by!(parent, child, dt)
    if child_failed(child)
        parent.force_stepfail = true
        return
    end

    @timeit_debug "sync <-" backward_sync_subintegrator!(parent, child, idxs, sync)
    return
end

function _require_two_operators(scheme, inner_algs)
    n = length(inner_algs)
    n == 2 || throw(
        ArgumentError(
            "$scheme is a two-operator (AB) table but got $n operators. Group the \
             operators into a nested GenericSplitFunction to use it with more."
        )
    )
    return nothing
end

# Every scheme prints as `NAME (inner -> inner)`; `separator` is what sits between the
# inner algorithms.
function _show_scheme(io::IO, name, inner_algs, separator = " -> ")
    print(io, name, " (")
    for inner_alg in inner_algs[1:(end - 1)]
        Base.show(io, inner_alg)
        print(io, separator)
    end
    length(inner_algs) > 0 && Base.show(io, inner_algs[end])
    return print(io, ")")
end
